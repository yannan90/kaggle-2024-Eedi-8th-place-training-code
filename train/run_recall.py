#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import json
import time
import os
import math
import sys
import pandas as pd
from torch import nn, Tensor
from tqdm.auto import tqdm, trange
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed, AutoConfig, AutoModel, MistralPreTrainedModel, MistralConfig, DynamicCache, \
    Cache
import random
import numpy as np
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    BitsAndBytesConfig
)
from transformers import optimization

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from utils.sft_dataset import is_rank_0
import torch.distributed as dist
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
# from transformers.models.mistral.modeling_flax_mistral import MISTRAL_INPUTS_DOCSTRING
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm
from transformers.utils import add_start_docstrings_to_model_forward

import random
# from peft import prepare_model_for_kbit_training
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from torch.utils.data import Dataset

IGNORE_INDEX = -100


def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8,
                        max_out_tokens=512,
                        enable_tensorboard=False,
                        enable_mixed_precision_lora=False,
                        tb_path="",
                        tb_name=""):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7,
        "memory_efficient_linear": False
    }
    if enable_mixed_precision_lora:
        zero_opt_dict["zero_quantized_nontrainable_weights"] = True
        zero_opt_dict["zero_hpz_partition_size"] = torch.cuda.device_count()
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE,
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/",
            "job_name": f"{tb_name}_tensorboard"
        }
    }


def get_optimizer_grouped_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora"],):
    optimizer_grouped_parameters = [
        
        # 第一组：需要权重衰减的参数，排除指定的 no_decay_name_list 和 lora_name_list 中的参数。
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and not any(nd in n
                                                    for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
        },
        
        # 第二组：属于 LoRA 层的参数，即参数名称包含 lora_name_list 中定义的关键字的参数，同时这些参数需要权重衰减，并应用一个单独的学习率 lora_lr。
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list)
                    and p.requires_grad and any(nd in n
                                                for nd in lora_name_list))
            ],
            "weight_decay":
                weight_decay,
            "lr":
                lora_lr
        },

        # 第三组：不需要权重衰减的参数，主要是 no_decay_name_list 中指定的参数。
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if not optimizer_grouped_parameters[1]["params"]:
        optimizer_grouped_parameters.pop(1)
    return optimizer_grouped_parameters


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def get_optimizer_only_lora_parameters(
        model,
        weight_decay,
        lora_lr=5e-4,
        no_decay_name_list=["bias", "LayerNorm.weight"],
        lora_name_list=["lora"],):  # 简化为只用 "lora" 匹配所有 LoRA 参数

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in lora_name_list) 
                and not any(nd in n for nd in no_decay_name_list)
            ],
            "weight_decay": weight_decay,
            "lr": lora_lr
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in lora_name_list)
                and any(nd in n for nd in no_decay_name_list)
            ],
            "weight_decay": 0.0,
            "lr": lora_lr
        },
    ]


    return optimizer_grouped_parameters
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


import inspect
import math
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union

import datasets
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizer

# ==============================
# data => dataset
# ==============================

class TrainDatasetForEmbedding(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer
    ):
        if os.path.isdir(args.train_data):
            train_datasets = []
            for file in os.listdir(args.train_data):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(args.train_data, file),
                                                     split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data, split='train')

        self.tokenizer = tokenizer
        self.args = args
        print("训练样本数量: ", len(self.dataset))

        # 加载misconception映射
        df_misc = pd.read_csv(f'{args.path_pre}/data/misconception_mapping.csv') # data
        self.misc_dict = dict(zip(df_misc['MisconceptionId'], df_misc['MisconceptionName']))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Tuple[str, List[str]]:

        '''
        返回 (query, misconceptions(里面放所有的misconcpetions, 第一个是pos，后面全是negs))
        '''
        
         # query: recall_query
        query = self.dataset[item]['query']
        if self.args.query_instruction_for_retrieval is not None:
            query = self.args.query_instruction_for_retrieval + query #加个引导前缀，使模型更清楚地知道每段文本的角色。

        # misconceptions 里面放所有的misconcpetions, 第一个是pos，后面全是negs
        misconceptions = []
        misconceptions.append(self.misc_dict[self.dataset[item]['pos']])
        if len(self.dataset[item]['neg']) < self.args.train_group_size - 1:
            num = math.ceil((self.args.train_group_size - 1) / len(self.dataset[item]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.args.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[item]['neg'], self.args.train_group_size - 1)
        misconceptions.extend([self.misc_dict[neg] for neg in negs])

        if self.args.misc_instruction_for_retrieval is not None:
            misconceptions = [self.args.misc_instruction_for_retrieval + p for p in misconceptions]
        return query, misconceptions


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_misc]] to List[qry], List[misc]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 512
    misc_max_len: int = 256

    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    # 随机掩码有 10% 的概率遮盖 input_ids 中的一部分 token，并将它们替换为特定值（这里是 2）。增强模型的能力。
    def mask_pad_token(self,q):
        if random.random()>0.9:
            tensor = q['input_ids'].float()
            # 创建一个与原始张量形状相同的随机张量
            mask = torch.rand(tensor.shape)

            # 设置阈值，将大于阈值的部分设置为1，小于阈值的部分设置为0
            mask = (mask > 0.9).float()

            # 使用mask张量将原始张量中的一部分元素设置为2
            tensor = tensor * (1 - mask) + 2 * mask
            tensor = tensor.long()
            q['input_ids'] = tensor
        return q


    def __call__(self, features):
        query = [f[0] for f in features]
        misc = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, []) # 将子列表逐一展开并合并到一个新的列表中。
        if isinstance(misc[0], list):
            misc = sum(misc, []) # 将子列表逐一展开并合并到一个新的列表中。

        q_collated = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        q_collated = self.mask_pad_token(q_collated)

        d_collated = self.tokenizer(
            misc,
            padding=True,
            truncation=True,
            max_length=self.misc_max_len,
            return_tensors="pt",
        )
        d_collated = self.mask_pad_token(d_collated)

        # return {"query": q_collated, "misc": d_collated}

        # 只返回必要的字段， 去掉labels
        return {
            "query": {
                "input_ids": q_collated["input_ids"],
                "attention_mask": q_collated["attention_mask"],
            },
            "misc": {
                "input_ids": d_collated["input_ids"],
                "attention_mask": d_collated["attention_mask"],
            }
        }



# ==============================
# modeling
# ==============================


class IgnoreLabelsWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        
        # 添加必要的方法
        self.prepare_inputs_for_generation = model.prepare_inputs_for_generation
        self.get_encoder = model.get_encoder if hasattr(model, 'get_encoder') else None

        
    def forward(self,labels=None, **kwargs,):
        return self.model(**kwargs)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)



# biencoder
class BiEncoderModel(nn.Module):
    def __init__(self,
                 args,
                 # base_model_related **********************************************************************
                 normlized: bool = True,
                 # base_model_related **********************************************************************
                 negatives_cross_device: bool = False,
                 temperature: float = 1.0, # 控制计算相似度时得分的“锐度”或“平滑度”，这种用法在对比学习（contrastive learning）中比较常见
                 use_inbatch_neg: bool = True,
                 # base_model_related **********************************************************************
                 sentence_pooling_method: str = "last" # 句子池化方法，可选 "mean", "cls", "last"
                 # base_model_related **********************************************************************
                 ):
        super().__init__()


        # 基础模型封装bnb和lora
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        loraConfig = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules.split(","),
            bias="none",
            lora_dropout=0.05,  # Conventional
            # task_type="CAUSAL_LM",
            task_type="FEATURE_EXTRACTION",
            inference_mode=False,  # 确保不是推理模式
        )
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        ### base_model for encoder
        # model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(args.model_name_or_path, trust_remote_code=True, quantization_config=bnb_config)
        model = get_peft_model(model, loraConfig)
        if args.lora_path !='none':
            print("加载之前的lora参数...", args.lora_path)
            d = torch.load(args.lora_path, map_location=model.device)
            model.load_state_dict(d, strict=False)
        self.model = IgnoreLabelsWrapper(model)
        
        # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        self.model.print_trainable_parameters()

        # # LLM for query augmentation
        # self.llm = AutoModelForCausalLM.from_pretrained(args.llm_model_name_or_path)

        # self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean',label_smoothing=0.2)
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config

        if not normlized:
            self.temperature = 1.0
            print("reset temperature = 1.0 due to using inner product to compute similarity")
        if normlized:
            if self.temperature > 0.5:
                raise ValueError(
                    "Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1")

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            #     logger.info("Run in a single GPU, set negatives_cross_device=False")
            #     self.negatives_cross_device = False
            # else:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()


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


    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)
    
    def last_token_pool(self, last_hidden_states: Tensor,
                        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == 'mean':
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == 'cls':
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == 'last':
            return self.last_token_pool(hidden_state, mask)

    def encode(self, features):
        if features is None:
            return None


        device = next(self.model.parameters()).device
        features = {
            'input_ids': features['input_ids'].to(device),
            'attention_mask': features['attention_mask'].to(device)
        }

        out = self.model(**features)
        out_em = self.sentence_embedding(out.last_hidden_state, features['attention_mask'])

        if self.normlized:
            out_em = torch.nn.functional.normalize(out_em, dim=-1)
        return out_em.contiguous()

    def compute_similarity(self, q_em, m_em):
        if len(m_em.size()) == 2:
            return torch.matmul(q_em, m_em.transpose(0, 1))
        return torch.matmul(q_em, m_em.transpose(-2, -1))

    def forward(self, query, misc):
        q_em = self.encode(query)
        m_em = self.encode(misc)

        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_em = self._dist_gather_tensor(q_em)
                m_em = self._dist_gather_tensor(m_em)

            group_size = m_em.size(0) // q_em.size(0) #得到每个query对应多少个misc

            if self.use_inbatch_neg:
                scores = self.compute_similarity(q_em, m_em) / self.temperature  # B, B*G
                scores = scores.view(q_em.size(0), -1) # B, B*G

                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long) #生成从 0 到 scores.size(0)-1 的整数序列，形状为 (B,)。
                target = target * group_size # 表示每个 query 的正样本在分组中的起始位置。
                loss = self.cross_entropy(scores, target)
            else:
                scores = self.compute_similarity(q_em[:, None, :, ],
                                                 m_em.view(q_em.size(0), group_size, -1)).squeeze(
                    1) / self.temperature

                scores = scores.view(q_em.size(0), -1)
                target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long) # 被设置为全零向量，表示每个分组的第一个样本是正样本。这意味着模型需要学习将分组中第一个样本（正样本）的得分设得比其他负样本高，从而达到排序的效果。
                loss = self.cross_entropy(scores, target)

        else:
            scores = self.compute_similarity(q_em, m_em)
            loss = None

        return dict(
            scores=scores,
            loss=loss,
            q_em=q_em,
            m_em=m_em,
        )


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# validation
from eedi_metrics import mapk

def inference(sentences, pids, model, tokenizer, device):
    batch_size = 32
    max_length = 512
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

    # 分批处理句子
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        with torch.no_grad():
            features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            features = to_device(features, device)
            embeddings = model.encode(features).detach()  
            embeddings = embeddings.cpu().numpy().tolist()
        all_embeddings.extend(embeddings)

    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result

def evaluation(valid_df, misc_df, model, tokenizer, device):

    model.eval()
    with torch.no_grad():

        query_embeddings = inference(list(valid_df['recall_query'].values), list(valid_df['QuestionId_Answer'].values), model, tokenizer, device)
        misc_embeddings = inference(list(misc_df['MisconceptionName'].values), list(misc_df['MisconceptionId'].values), model, tokenizer, device)
    
    # Ret_topNids = util.semantic_search(query_embeddings, misc_embeddings, top_k=25)
    # valid_df["predicted"] = [[int(x["corpus_id"]) for x in top25id] for top25id in Ret_topNids]
    
    # 提取文档嵌入并生成索引映射
    misc_index_map = {index: doc_id for index, doc_id in enumerate(list(misc_embeddings.keys()))}
    misc_embeddings = np.concatenate([e.reshape(1, -1) for e in list(misc_embeddings.values())])

    predicts = []
    for _, row in tqdm(valid_df.iterrows()):
        query_id = row['QuestionId_Answer']
        query_em = query_embeddings[query_id].reshape(1, -1)
    
        cosine_similarity = np.dot(query_em, misc_embeddings.T).flatten()
    
        sort_index = np.argsort(-cosine_similarity)[:25]
        pids = [misc_index_map[index] for index in sort_index]
        predicts.append(pids)

    truths = [[data] for data in valid_df["MisconceptionId"].values]
    map25 = mapk(truths, predicts)
    recall25 = 0
    for true_ids, pred_ids in zip(truths, predicts):
        if any(tid in pred_ids for tid in true_ids):
            recall25 += 1
    recall25 = recall25 / len(valid_df)
    return map25, recall25
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


def main():

    # ==============================
    # 并行配置
    # ==============================

    print(sys.executable)  # 打印当前环境的 Python 解析器目录
    args = parse_args()

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # local_rank 参数决定使用哪个GPU。如果 local_rank 为 -1，则使用第一个可用的GPU
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    # 3 当前进程的全局排名
    args.global_rank = torch.distributed.get_rank()
    # DeepSpeed配置: 获取训练配置，设置批次大小和其他训练参数。
    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step1_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()


    # ==============================
    # 训练准备
    # ==============================


    #### 分词器和模型加载
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = BiEncoderModel(args, normlized=True, negatives_cross_device=True, temperature=0.02)

    train_dataset = TrainDatasetForEmbedding(args, tokenizer)

    #### DataLoaders creation:
    if args.local_rank == -1: #这是单机或单 GPU， 随机打乱
        train_sampler = RandomSampler(train_dataset)
    else: #将数据划分给每个 GPU，使得每个进程只处理数据集的一部分，确保多个进程不会重复采样数据
        train_sampler = DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset,
                                  shuffle=(train_sampler is None),
                                  collate_fn=EmbedCollator(
                                      tokenizer,
                                      query_max_len=args.query_max_len,
                                      misc_max_len=args.misc_max_len
                                  ),
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size,
                                  pin_memory=True)


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # load 全部misc和valid_df，做validation用
    if args.CV_fold>0:
        misc_df = df = pd.read_csv(f'{args.path_pre}/data/misconception_mapping.csv')
        valid_df = pd.read_pickle(f"{args.path_pre}/data/valid_{args.CV_fold}_{args.valid_fold}.pkl")
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


    #### 设置需要微调的参数以及学习率
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # # 全模型微调
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    #     model, args.weight_decay, args.lora_learning_rate)
    
    # 仅微调lora参数
    optimizer_grouped_parameters = get_optimizer_only_lora_parameters(
        model, args.weight_decay, args.lora_learning_rate)
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps)

    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=math.ceil(max_steps * 0.03) if args.num_warmup_steps == 0 else args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    

    # ==============================
    # 训练开始
    # ==============================

    # 首次验证
    print_rank_0("***** INITIAL EVALUATION *****", args.global_rank)
    val_map25, val_recall25 = evaluation(valid_df, misc_df, model, tokenizer, device)
    print_rank_0(f"Valid map25: {val_map25}, Hit Rate in Top 25: {val_recall25}", args.global_rank)

    print_rank_0("***** Running training *****", args.global_rank)
    print_rank_0(f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****", args.global_rank)

    total_steps = len(train_dataloader) * args.num_train_epochs
    total_loss = 0.
    best_val_map25 = 0. #记录最优验证参数
    best_val_recall25 = 0. #记录最优验证参数
    no_improve_epoch = 0.
    global_step = -1
    time_start = time.time()
    cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Batches {len(train_dataloader)}",
            args.global_rank)
        
        for step, batch in enumerate(train_dataloader):
            model.train()
            global_step += 1
            batch = to_device(batch, device)
            loss = model(**batch)['loss']
            model.backward(loss)
            model.step()
            total_loss += loss.item()

        ### log & save

            if global_step % 10 == 0: #每10步打印一次日志的条件判断。
                time_end = time.time()
                total_time = time_end - time_start  # 计算运行总时间
                time_start = time_end
                print_rank_0(
                    f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, curr_step:{global_step}/{total_steps} curr_loss {loss.item()} lr:{lr_scheduler.get_last_lr()[0]} use time:{total_time}s",
                    args.global_rank)
            if (global_step + 1) % args.gradient_accumulation_steps == 0: #每个梯度累积周期结束时（即 gradient_accumulation_steps 达到后）重置 total_loss
                loss_mean = total_loss / args.gradient_accumulation_steps  # calculate mean loss
                total_loss = 0.
            
            # 当 args.save_batch_steps 被设置，且当前步数达到指定的间隔时，在所有进程中都进行验证和保存
            if args.save_batch_steps and (global_step + 1) % args.save_batch_steps == 0 and (is_rank_0() or args.zero_stage == 3):

                print_rank_0(f"***** Evaluating Loss, Epoch {epoch + 1}/{args.num_train_epochs}---Step {global_step}/{total_steps}*****",args.global_rank)
                print_rank_0(f"loss: {loss_mean}", args.global_rank)
                    
                if args.CV_fold>0:
                    # 调用 evaluation 计算验证指标
                    val_map25, val_recall25 = evaluation(valid_df, misc_df, model, tokenizer, device)
                    print_rank_0(f"Valid map25: {val_map25}, Hit Rate in Top 25: {val_recall25}", args.global_rank)
                    
                    if val_map25 > best_val_map25:
                        print_rank_0(f"val_log----epoch:{epoch},batch:{global_step + 1},save model from {best_val_map25} to {val_map25} !!!",args.global_rank)
                        save_model(args, model, tokenizer, f"best_val_map25_model")
                        best_val_map25 = val_map25
                        no_improve_epoch = 0
                    else:
                        no_improve_epoch += 1
                        print_rank_0(f"val_log----epoch:{epoch},batch:{global_step + 1},no_improve_epoch:{no_improve_epoch},curr_val_map25 {val_map25} best_val_map25 {best_val_map25} ...", args.global_rank)
                    if args.earystop and no_improve_epoch == args.eary_stop_epoch:
                        print_rank_0( f"val_log----epoch:{epoch},batch:{global_step + 1} eary stop,best_val_map25 {best_val_map25} !!!",args.global_rank)
                        return
                    
                    # if val_recall25 > best_val_recall25:
                    #     print_rank_0(f"val_log----epoch:{epoch},batch:{global_step + 1},save model from {best_val_recall25} to {val_recall25} !!!",args.global_rank)
                    #     save_model(args, model, tokenizer, f"best_val_recall25_model")
                    #     best_val_recall25 = val_recall25
                    #     no_improve_epoch = 0
                    # else:
                    #     no_improve_epoch += 1
                    #     print_rank_0(f"val_log----epoch:{epoch},batch:{global_step + 1},no_improve_epoch:{no_improve_epoch},curr_val_recall25 {val_recall25} best_val_recall25 {best_val_recall25} ...", args.global_rank)
                    # if args.earystop and no_improve_epoch == args.eary_stop_epoch:
                    #     print_rank_0( f"val_log----epoch:{epoch},batch:{global_step + 1} eary stop,best_val_recall25 {best_val_recall25} !!!",args.global_rank)
                    #     return
        # 保存每一轮
        if args.save_per_epoch == 1:
            save_model(args, model, tokenizer, f"epoch_{epoch}_model")
        # 保存最后一轮
        if epoch == args.num_train_epochs - 1:
            save_model(args, model, tokenizer, f"epoch_{epoch}_model")
        model.tput_timer.update_epoch_count()
    
    


def save_model(args, model, tokenizer, sub_folder=None): #只保存tokenizer和lora权重
    if sub_folder is not None:
        output_dir = os.path.join(args.output_dir, sub_folder)
        print_rank_0('saving model ...', args.global_rank)
        
        # 保存 tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # 仅在 global_rank 为 0 的主进程上保存模型
        if args.global_rank == 0:
            
            model_to_save = model.module if hasattr(model, 'module') else model # 获取模型的实际结构（考虑到多卡情况）
            model_to_save = model_to_save.model

            # 检查并解包 IgnoreLabelsWrapper
            if isinstance(model_to_save, IgnoreLabelsWrapper):
                model_to_save = model_to_save.model

            # 提取并保存LoRA相关的权重
            lora_weights = {k: v for k, v in model_to_save.state_dict().items() if "lora" in k}
            output_lora_file = os.path.join(output_dir, "adapter.bin")
            torch.save(lora_weights, output_lora_file)

        print_rank_0('saving success ...', args.global_rank)

def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")

    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")

    parser.add_argument('--save_batch_steps', type=int, default=1000)
    parser.add_argument('--earystop', type=bool, default=False)
    parser.add_argument('--eary_stop_epoch', type=int, default=2)
    parser.add_argument('--save_per_epoch', type=int, default=-1)
    parser.add_argument('--path_pre', type=str, default=None)
    parser.add_argument('--train_data', type=str, default=None, help="train data path ")
    parser.add_argument('--CV_fold', type=int, default=0)
    parser.add_argument('--valid_fold', type=int, default=0)
    parser.add_argument('--query_instruction_for_retrieval', type=str, default=None,
                        help="query_instruction_for_retrieval")
    parser.add_argument('--misc_instruction_for_retrieval', type=str, default=None,
                        help="misc_instruction_for_retrieval")
    parser.add_argument('--query_max_len', type=int, default=512)
    parser.add_argument('--misc_max_len', type=int, default=256)
    parser.add_argument('--train_group_size', type=int, default=2, help="train_group_size")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=0,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_path", type=str, default='none')
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_config",
                        type=str,
                        default="./configs/lora_config_llama.json",
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument('--lora_r',
                        type=int,
                        default=64)
    parser.add_argument('--lora_alpha',
                        type=int,
                        default=128)
    parser.add_argument('--lora_target_modules',
                        type=str,
                        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")


    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert (
            not args.only_optimize_lora
        ), "--gradient_checkpointing and --only_optimize_lora cannot be enabled at the same time."

    return args

if __name__ == "__main__":
    main()
