import logging
from dataclasses import dataclass
from typing import Dict, Optional, List, Union

import torch
from torch import nn, Tensor
from transformers import AutoTokenizer
from transformers.file_utils import ModelOutput

logger = logging.getLogger(__name__)

@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    logits: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    def __init__(self,
                 model: None,
                 tokenizer: AutoTokenizer = None,
                 train_batch_size: int = 4,
                 ):
        super().__init__()



        self.model = model
        self.tokenizer = tokenizer
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.1)
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.config = self.model.config

        self.train_batch_size = train_batch_size

        self.yes_loc = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]

    # 😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀
    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
            
        # 确保输入是连续的
        t = t.contiguous()
        
        # 收集所有进程的张量
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        torch.distributed.all_gather(all_tensors, t)
        
        # 将所有张量连接起来
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        
        return all_tensors
    # 😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀😀


    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, features):
        if features is None:
            return None
        outputs = self.model(input_ids=features['input_ids'],
                             attention_mask=features['attention_mask'],
                             position_ids=features['position_ids'] if 'position_ids' in features.keys() else None,
                             output_hidden_states=True) # (B, Len, Vocab)
        _, max_indices = torch.max(features['labels'], dim=1) # (B,) # 每个样本的最大类别索引
        predict_indices = max_indices - 1 # (B,)
        logits = [outputs.logits[i, predict_indices[i], :] for i in range(outputs.logits.shape[0])]
        logits = torch.stack(logits, dim=0) # (B, num_vocab) # 每个样本的关键点位的预测词分布
        scores = logits[:, self.yes_loc] # (B, 1) # 每个样本的关键点位的yes预测值

        return scores.contiguous()



    def forward(self, pair: Union[Dict[str, Tensor], List[Dict[str, Tensor]]] = None):  #DataLoader可以在模型forward中自行解包，根据键名获取collated
        ranker_logits = self.encode(pair) # (B*G，1)

        if self.training:
            grouped_logits = ranker_logits.view(self.train_batch_size, -1) # (B，G*1)
            target = torch.zeros(self.train_batch_size, device=grouped_logits.device, dtype=torch.long) # (B,) # 被设置为全零向量，表示每个分组的第一个样本是正样本。这意味着模型需要学习将分组中第一个样本（正样本）的得分设得比其他负样本高，从而达到排序的效果。
            loss = self.compute_loss(grouped_logits, target) #标量
        else:
            loss = None
        return RerankerOutput(loss=loss,logits=ranker_logits,)

    def compute_loss(self, logits, target):
        return self.cross_entropy(logits, target)

    def save(self, output_dir: str):

        self.tokenizer.save_pretrained(output_dir)

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model # 获取模型的实际结构（考虑到多卡情况）

        lora_weights = {k: v.clone().cpu() for k,v in model_to_save.state_dict().items() if "lora" in k}

        output_lora_file = f"{output_dir}/adapter.bin"
        torch.save(lora_weights, output_lora_file)
