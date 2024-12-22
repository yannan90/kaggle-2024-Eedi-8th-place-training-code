import re
import sys
from typing import List

import math
import os.path
import random
from dataclasses import dataclass

import datasets
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizer, BatchEncoding

from arguments import DataArguments


class TrainDatasetForReranker(Dataset):
    def __init__(
            self,
            data_files,
            tokenizer,
            query_max_len = 512,
            misc_max_len = 128,
            train_group_size = 8,
            path_pre = ""
    ):
        self.dataset = datasets.load_dataset('json', data_files=data_files, split='train')

        
        self.tokenizer = tokenizer
        self.total_len = len(self.dataset)
        self.max_length = query_max_len + misc_max_len
        self.train_group_size = train_group_size
        self.misc_contents = list(pd.read_csv(f'{path_pre}/data/misconception_mapping.csv')['MisconceptionName'].values)

    def __len__(self):
        return self.total_len


    def __getitem__(self, idx) -> List[BatchEncoding]:
        '''
        返回一个list，里面每个元素是一个字典，包含： 'input_ids', 'attention_mask', 'labels'

        每个序列内容：query + misc + 俩回车 + prompt + yes
        '''

        # misconceptions 里面放所有的misconcpetions, 第一个是pos，后面全是negs
        misconceptions = []
        pos = self.dataset[idx]['pos']
        misconceptions.append(pos)
        if len(self.dataset[idx]['neg']) < self.train_group_size - 1:
            num = math.ceil((self.train_group_size - 1) / len(self.dataset[idx]['neg']))
            negs = random.sample(self.dataset[item]['neg'] * num, self.train_group_size - 1)
        else:
            negs = random.sample(self.dataset[idx]['neg'], self.train_group_size - 1)
        misconceptions.extend(negs)
        misconceptions = [self.misc_contents[m] for m in misconceptions] #加个引导前缀，使模型更清楚地知道每段文本的角色。

        yes_input_ids = self.tokenizer('Yes', return_tensors=None, add_special_tokens=False)['input_ids']

        queries_inputs = []
        for i, misconception in enumerate(misconceptions):
            msg = [
                {"role": "system", "content": "You are a Mathematics teacher. "},
                {"role": "user", "content": self.dataset[idx]['query'].rstrip()+ " " + misconception + "\n\nPlease respond with only 'Yes' or 'No'."}
            ]
            query = self.tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            print(query)
            query_encoded = self.tokenizer(query,
                                          return_tensors=None,
                                          max_length=self.max_length - len(yes_input_ids), #给yes留出位置
                                          truncation=True,
                                          add_special_tokens=False)

            query_input_ids = query_encoded['input_ids']

            # 添加 BOS 标记（如果存在且不等于 PAD）
            if self.tokenizer.bos_token_id is not None and self.tokenizer.bos_token_id != self.tokenizer.pad_token_id:
                query_input_ids = [self.tokenizer.bos_token_id] + query_input_ids

            # 准备模型输入
            prepared_input = self.tokenizer.prepare_for_model(
                query_input_ids,
                truncation=True,
                max_length=self.max_length - len(yes_input_ids), #给yes留出位置
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False
            )


            query_input_ids = prepared_input['input_ids'] + yes_input_ids

            batch = {
                'input_ids': query_input_ids,
                'attention_mask': [1] * len(query_input_ids),  # 设为与 input_ids 长度相同的全 1 向量，表示输入中的所有 token 都是有效的，不需要遮掩。
                'labels': [-100] * (len(query_input_ids) - 1) + [query_input_ids[-1]] # 将除最后一个 token 外的位置都设置为 -100，这样模型在计算损失时只会关注最后一个 token 的预测。
            }

            queries_inputs.append(batch)

        return queries_inputs


@dataclass
class RerankCollator(DataCollatorForSeq2Seq):
    """
    数据整理器
    拆分和整理：将输入的查询（query）和候选样本（passage）整理成模型所需的批次格式。
    填充序列：将长度不同的 input_ids 和 labels 填充到相同的长度
    批次化处理：生成一个包含查询和候选样本对的批次（batch），将其转换为张量，并返回给模型进行训练。
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 512
    misc_max_len: int = 256


    def __call__(self, features, return_tensors='pt'):
        if return_tensors is None:
            return_tensors = self.return_tensors

        # 保证输入是一个一维的list，剥掉batch维度
        if isinstance(features[0], list):
            features = sum(features, [])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None

        
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # 把labels填充到统一长度，就这么简单，没干别的。因为labels 不会被 tokenizer.pad 自动填充
        if labels is not None:
            max_label_length = max(len(l) for l in labels) # 计算 labels 中最长序列的长度，用于填充到统一长度。
            # print(max_label_length)
            if self.pad_to_multiple_of is not None: # 如果指定了 pad_to_multiple_of，将 max_label_length 调整为指定的倍数。确保填充后的序列长度对齐到指定倍数，通常用于加速计算。

                max_label_length = (
                        (max_label_length + self.pad_to_multiple_of - 1)
                        // self.pad_to_multiple_of
                        * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side # 获取填充方向（左右）。

            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        # 统一填充到相同的长度
        collated = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.query_max_len + self.misc_max_len,
            return_tensors=return_tensors,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {"pair": collated} #DataLoader可以在模型forward中自行解包，获取collated
        # return collated

