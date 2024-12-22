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


def inference(sentences, pids, model, tokenizer, batch_size, max_length, sentence_pooling_method, device):
    all_embeddings = []
    length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
    sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
    for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=False):
        sentences_batch = sentences_sorted[start_index: start_index + batch_size]
        features = tokenizer(sentences_batch, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
        features = batch_to_device(features, device)
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



if __name__ == '__main__':

    
    path_pre = sys.argv[1]
    model_version = sys.argv[2]
    model_path = sys.argv[3]
    train_data = sys.argv[4]
    lora_path = sys.argv[5]
    lora_r = int(sys.argv[6])
    lora_alpha = int(sys.argv[7])
    lora_target_modules = sys.argv[8]
    CV_fold = int(sys.argv[9])
    valid_fold = int(sys.argv[10])
    num_gpu = int(sys.argv[11])
    infer_batch = int(sys.argv[12])
    infer_max_len = int(sys.argv[13])
    sentence_pooling_method = sys.argv[14]
    neg_cnt_1 = int(sys.argv[15])
    neg_cnt_2 = int(sys.argv[16])

    loraConfig = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules.split(","),
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
    

    # =============================
    # encoding misconceptions
    # =============================

    df = pd.read_csv(f'{path_pre}/data/misconception_mapping.csv') # data

    device = 'cuda:0'

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map=device)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, quantization_config=bnb_config, device_map=device)
    if lora_path !='none':
        print("加载之前的lora参数...", lora_path)
        model = get_peft_model(model, loraConfig)
        d = torch.load(lora_path, map_location=model.device)
        model.load_state_dict(d, strict=False)
        # model = model.merge_and_unload()
    model = model.eval()
    # model = model.to(device)

    sentences = list(df['MisconceptionName'].values)
    pids = list(df['MisconceptionId'].values)
    results = inference(sentences, pids, model, tokenizer, infer_batch , infer_max_len, sentence_pooling_method, device)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    

    with open(f"{path_pre}/data/{model_version}_misc.pkl", 'wb') as f:
        pickle.dump(results, f)



    # =============================
    # encoding query
    # =============================


    df = pd.read_pickle(f'{path_pre}/data/{train_data}')

    # df = df.head(200) # testing only

    # CV split
    if CV_fold>0:
        df_val= df[df[f"{CV_fold}_fold"]==valid_fold]
        df = df[df[f"{CV_fold}_fold"]!=valid_fold]

        df_val.to_pickle(f"{path_pre}/data/valid_{CV_fold}_{valid_fold}.pkl")
    
    print("train_data length: ", len(df))
    print("valid_data length: ", len(df_val))


    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # 利用多线程和多GPU并行推理来提升处理速度。
    # use_device = [f'cuda:{i}' for i in range(num_gpu)]
    
    use_device = ['cuda:0','cuda:1'] if num_gpu>1 else ['cuda:0']
    print("线程数: ", len(use_device))

    
    from threading import Thread

    # 每个device上都放一个模型
    models = []
    for device in use_device:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device_map=device)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True, quantization_config=bnb_config, device_map=device)
        if lora_path !='none':
            print("加载之前的lora参数...", lora_path)
            model = get_peft_model(model, loraConfig)
            d = torch.load(lora_path, map_location=model.device)
            model.load_state_dict(d, strict=False)
            # model = model.merge_and_unload() 
        model = model.eval()
        # model = model.to(device)
        models.append((model, tokenizer))


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # embed train folds
    query_embedding_dict = {}
    def run_inference(df, model, index, device):
        sentences = list(df['recall_query'].values)
        pids = list(df['QuestionId_Answer'].values)
        query_embedding_dict.update(inference(sentences, pids, model[0], model[1], infer_batch, infer_max_len, sentence_pooling_method, device))

    # 按 use_device 数量将 df 切分为几部分，分别分配给每个设备，并用多线程并行推理处理，加速计算。
    df['divide'] = list(range(len(df)))
    df['divide'] = df['divide'] % len(use_device)

    # 这里使用多线程将 df 按 fold 划分，每个线程在不同的 GPU 上运行 run_inference，即每个线程生成一部分论文的嵌入并更新到 results 中。
    ts = []
    for index, device in enumerate(use_device):
        t0 = Thread(target=run_inference, args=(df[df['divide'] == index], models[index], index, device))
        ts.append(t0)
    for i in range(len(ts)):
        ts[i].start()
    for i in range(len(ts)):
        ts[i].join()


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # embed valid folds
    query_embedding_dict_val = {}
    def run_inference(df, model, index, device):
        sentences = list(df['recall_query'].values)
        pids = list(df['QuestionId_Answer'].values)
        query_embedding_dict_val.update(inference(sentences, pids, model[0], model[1], infer_batch, infer_max_len, sentence_pooling_method, device))

    # 按 use_device 数量将 df 切分为几部分，分别分配给每个设备，并用多线程并行推理处理，加速计算。
    df_val['divide'] = list(range(len(df_val)))
    df_val['divide'] = df_val['divide'] % len(use_device)

    # 这里使用多线程将 df 按 fold 划分，每个线程在不同的 GPU 上运行 run_inference，即每个线程生成一部分论文的嵌入并更新到 results 中。
    ts = []
    for index, device in enumerate(use_device):
        t0 = Thread(target=run_inference, args=(df_val[df_val['divide'] == index], models[index], index, device))
        ts.append(t0)
    for i in range(len(ts)):
        ts[i].start()
    for i in range(len(ts)):
        ts[i].join()
    

    del models, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    # =============================
    # output train data
    # =============================

    device = 'cuda:0'

    # 读取misc_embeddings
    with open(f"{path_pre}/data/{model_version}_misc.pkl", 'rb') as f:
        misc_embedding_dict = pickle.load(f)
    misc_embeddings = np.concatenate([e.reshape(1, -1) for e in list(misc_embedding_dict.values())])
    misc_embeddings_tensor = torch.tensor(misc_embeddings).to(device)
    index_to_misc_index = {index: misc_id for index, misc_id in enumerate(list(misc_embedding_dict.keys()))}
    
    # 预测一轮 for train
    misc_preds = []
    for _, row in tqdm(df.iterrows()):
        query_id = row['QuestionId_Answer']
        query_em = query_embedding_dict[query_id].reshape(1, -1)
        query_em = torch.tensor(query_em).to(device).view(1, -1)
        score = F.cosine_similarity(query_em, misc_embeddings_tensor)
        sort_index = torch.sort(-score).indices.detach().cpu().numpy().tolist()[:1000]
        pids = [index_to_misc_index[index] for index in sort_index]
        misc_preds.append(pids)
    df['top_recall_pids'] = misc_preds
 
    # 预测一轮 for valid
    misc_preds = []
    for _, row in tqdm(df_val.iterrows()):
        query_id = row['QuestionId_Answer']
        query_em = query_embedding_dict_val[query_id].reshape(1, -1)
        query_em = torch.tensor(query_em).to(device).view(1, -1)
        score = F.cosine_similarity(query_em, misc_embeddings_tensor)
        sort_index = torch.sort(-score).indices.detach().cpu().numpy().tolist()[:1000]
        pids = [index_to_misc_index[index] for index in sort_index]
        misc_preds.append(pids)
    df_val['top_recall_pids'] = misc_preds

    # ==================================================
    ### 生成 hard neg mining 的训练数据

    def filterout_positive(x, y):
        return [xi for xi in x if xi != y]

    recall_data = df
    recall_data['new_hard_recall_pids'] = recall_data.apply(
        lambda row: filterout_positive(row['top_recall_pids'], row['MisconceptionId']), axis=1
    )

    print("train_data for recall length: ", len(recall_data))

    # 提取前200个hard neg recalls, 用来训练recall model
    # 只保存query, pos_misc_id, neg_misc_id
    cnt = neg_cnt_1
    next_train = []
    for _, row in recall_data.iterrows():
        query = row['recall_query']
        pos = int(row['MisconceptionId'])  # 转换为 int 类型
        negs = [int(neg) for neg in row['new_hard_recall_pids'][:cnt]] 
        next_train.append({'query': query, 'pos': pos, 'neg': negs})
    with open(f"{path_pre}/data/{model_version}_recall_top_{cnt}.jsonl", 'w') as f:
        json.dump(next_train, f)

    # ==================================================
    rerank_data = pd.concat([df, df_val], ignore_index=True)
    rerank_data = rerank_data[~rerank_data['QuestionId_Answer'].astype(str).str.endswith(('0','1','2','3','4','5'))]
    rerank_data['new_hard_recall_pids'] = rerank_data.apply(
        lambda row: filterout_positive(row['top_recall_pids'], row['MisconceptionId']), axis=1
    )

    print("train_data for rerank length: ", len(rerank_data))

    # 提取前100个hard neg recalls, 用来训练rerank model
    cnt = neg_cnt_2
    next_train = []
    for _,row in rerank_data.iterrows():
        query = row['rerank_query']
        pos = int(row['MisconceptionId'])  # 转换为 int 类型
        negs = [int(neg) for neg in row['new_hard_recall_pids'][:cnt]] 
        next_train.append({'query':query,'pos':pos,'neg':negs,'prompt':"Please respond with only 'Yes' or 'No'."})
    with open(f"{path_pre}/data/{model_version}_recall_top_{cnt}_for_rerank.jsonl", 'w') as f:
        json.dump(next_train, f)


