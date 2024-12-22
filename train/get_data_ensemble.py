import pandas as pd
import json
import pickle
from bs4 import BeautifulSoup
import numpy as np
from tqdm.auto import tqdm
from tqdm.autonotebook import trange

import itertools

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    # prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import os, gc, sys
import warnings
warnings.filterwarnings('ignore')


def batch_to_device(batch, target_device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

def sentence_embedding(hidden_state, mask, sentence_pooling_method):
    if sentence_pooling_method == 'mean':
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif sentence_pooling_method == 'cls':
        return hidden_state[:, 0]
    elif sentence_pooling_method == 'last':
        return last_token_pool(hidden_state, mask)


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def inference(sentences, pids, model, tokenizer, batch_size, max_length, sentence_pooling_method):
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        features = batch_to_device(features, model.device)
        with torch.no_grad():
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            outputs = model(**features)
            embeddings = sentence_embedding(outputs.last_hidden_state, features['attention_mask'], sentence_pooling_method)
            embeddings = torch.nn.functional.normalize(embeddings , dim=-1)
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            embeddings = embeddings.detach().cpu().numpy().tolist()
        all_embeddings.extend(embeddings)

    all_embeddings = [np.array(all_embeddings[idx]).reshape(1, -1) for idx in np.argsort(length_sorted_idx)]

    sentence_embeddings = np.concatenate(all_embeddings, axis=0)
    result = {pids[i]: em for i, em in enumerate(sentence_embeddings)}
    return result


# 😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️
# 😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️
# ### 多线程推理
from threading import Thread
from queue import Queue

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# 并行推理 helper funcs
def run_inference(sentences, pids, model, batch_size, max_length, sentence_pooling_method, result_queue, index):
    result = inference(sentences, pids, model[0], model[1], batch_size, max_length, sentence_pooling_method)
    result_queue.put((index, result))  # 将线程索引与结果一起存入队列


# 😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️
# 😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️😶‍🌫️


if __name__ == '__main__':

    path_pre = sys.argv[1]
    model_version = sys.argv[2]
    model_path = sys.argv[3]
    train_data = sys.argv[4]
    lora_paths = sys.argv[5].split(",")
    weights = [float(x) for x in sys.argv[6].split(",")]
    num_gpu = int(sys.argv[7])
    infer_batch = int(sys.argv[8])
    infer_max_len = int(sys.argv[9])
    sentence_pooling_method = sys.argv[10]
    neg_cnt = int(sys.argv[11])

    loraConfig = LoraConfig(
                r=32,
                lora_alpha=64,
                target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
                bias="none",
                lora_dropout=0.05,  # Conventional
                task_type="FEATURE_EXTRACTION",
            )

    # BitsAndBytesConfig 用于配置模型的量化参数，以便在使用较少的内存和计算资源的情况下运行模型：
    bnb_config = BitsAndBytesConfig( 
            load_in_4bit=True, #将模型权重加载为4位格式，减少内存占用。
            bnb_4bit_use_double_quant=True, #使用双重量化方法来提高模型的性能。
            bnb_4bit_quant_type="nf4", #指定量化类型为 nf4（一个特定的量化格式）。
            bnb_4bit_compute_dtype=torch.bfloat16 #指定计算的数据类型为 bfloat16，通常用于提升计算速度和减小内存占用。
        )
    

    use_device = [f'cuda:{i}' for i in range(torch.cuda.device_count())]

    models=[]
    for i in range(len(lora_paths)):

        path=lora_paths[i]
        device=use_device[i%len(use_device)]

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, quantization_config=bnb_config, device_map=device)
        if path !='none':
            print("加载之前的lora参数...", path)
            model = get_peft_model(model, loraConfig)
            d = torch.load(path, map_location=model.device)
            model.load_state_dict(d, strict=False)
        model = model.eval()
        models.append((model, tokenizer))


    # =============================
    # encoding misconceptions
    # =============================

    df = pd.read_csv(f'{path_pre}/data/misconception_mapping.csv') # data

    sentences = list(df['MisconceptionName'].values)
    pids = list(df['MisconceptionId'].values)

    result_queue = Queue()  # 单一队列存储结果，带索引确保顺序
    threads = []
    for model_index, model in enumerate(models):
        t = Thread(target=run_inference,args=(sentences, pids, model, infer_batch, infer_max_len, sentence_pooling_method, result_queue, model_index))
        threads.append(t)
    for thread in threads: # 启动线程
        thread.start()
    for thread in threads: # 等待所有线程完成
        thread.join()

    # 收集每个线程的结果并按索引排序
    misc_embeds = sorted([result_queue.get() for _ in threads], key=lambda x: x[0])
    misc_embeds = [result[1] for result in misc_embeds]  # 每个元素是dict: {pids[i]: em for i, em in enumerate(sentence_embeddings)}

    # =============================
    # encoding query
    # =============================

    df = pd.read_pickle(f'{path_pre}/data/{train_data}')
    # df = df.head(200) # testing only
    
    print("train_data length: ", len(df))

    sentences = list(df['recall_query'].values)
    pids = list(df['QuestionId_Answer'].values)

    result_queue = Queue()  # 单一队列存储结果，带索引确保顺序
    threads = []
    for model_index, model in enumerate(models):
        t = Thread(target=run_inference,args=(sentences, pids, model, infer_batch, infer_max_len, sentence_pooling_method, result_queue, model_index))
        threads.append(t)
    for thread in threads: # 启动线程
        thread.start()
    for thread in threads: # 等待所有线程完成
        thread.join()

    # 收集每个线程的结果并按索引排序
    qury_embeds = sorted([result_queue.get() for _ in threads], key=lambda x: x[0])
    qury_embeds = [result[1] for result in qury_embeds] # 每个元素是dict: {pids[i]: em for i, em in enumerate(sentence_embeddings)}


    del models, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # ====================================
    # 获得高排序样本
    # ====================================

    device = 'cuda:0'

    misc_preds = []
    for _, row in tqdm(df.iterrows()):
        query_id = row['QuestionId_Answer']
        final_scores = None  # Placeholder for accumulating weighted scores
        for i in range(len(qury_embeds)):
            query_em = qury_embeds[i][query_id].reshape(1, -1)
            query_em = torch.tensor(query_em).to(device).view(1, -1)

            misc_em = np.concatenate([e.reshape(1, -1) for e in list(misc_embeds[i].values())])
            misc_em = torch.tensor(misc_em).to(device)
            index_to_misc_index = {index: misc_id for index, misc_id in enumerate(list(misc_embeds[i].keys()))}

            # Calculate cosine similarity
            scores = F.cosine_similarity(query_em, misc_em)

            # Multiply scores by weight for the current i
            weighted_scores = scores * weights[i]

            # Accumulate scores
            final_scores = weighted_scores if final_scores is None else final_scores + weighted_scores

        # Sort the final accumulated scores and get the top 1000 indices
        sort_index = torch.sort(-final_scores).indices.detach().cpu().numpy().tolist()[:1000]

        # Map sorted indices back to misconception IDs
        pids = [index_to_misc_index[index] for index in sort_index]
        misc_preds.append(pids)
    df['top_recall_pids'] = misc_preds

    # ====================================
    # 生成 hard neg mining 的训练数据
    # ====================================

    rerank_data = df
    rerank_data = rerank_data[~rerank_data['QuestionId_Answer'].astype(str).str.endswith(('0','1','2','3','4','5'))]

    def filterout_positive(x, y):
        return [xi for xi in x if xi != y]

    rerank_data['new_hard_recall_pids'] = rerank_data.apply(
        lambda row: filterout_positive(row['top_recall_pids'], row['MisconceptionId']), axis=1
    )

    print("train_data for rerank length: ", len(rerank_data))


    # ====================================
    # 提取前neg_cnt个hard neg recalls, 用来训练rerank model
    # ====================================

    cnt = neg_cnt
    next_train = []
    for _,row in rerank_data.iterrows():
        query = row['rerank_query']
        pos = int(row['MisconceptionId'])  # 转换为 int 类型
        negs = [int(neg) for neg in row['new_hard_recall_pids'][:cnt]] 
        next_train.append({'query':query,'pos':pos,'neg':negs,'prompt':"Please respond with only 'Yes' or 'No'."})
    with open(f"{path_pre}/data/{model_version}_recall_top_{cnt}_for_rerank.jsonl", 'w') as f:
        json.dump(next_train, f)