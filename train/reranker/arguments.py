import os
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."}
    )
    target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,lm_head",
        metadata={"help": "Lora modules to apply LoRA to."}
    )
    lora_path: str = field(
        default="none",
        metadata={"help": "Previous lora model path."}
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "If passed, will use flash attention to train the model."}
    )


    cache_dir: str = field(
        default="tmp", metadata={"help": "the cache of the model"}
    )
    token: str = field(
        default=None, metadata={"help": "the token to access the huggingface model"}
    )


@dataclass
class DataArguments:
    
    path_pre: str = field(
        default="",
        metadata={"help": "path pre."}
    )
    train_data: str = field(
        default='toy_finetune_data.jsonl', metadata={"help": "Path to train data"}
    )

    valid_data: str = field(
        default='toy_finetune_data.jsonl', metadata={"help": "Path to valid data"}
    )

    train_group_size: int = field(default=8)

    query_max_len: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    misc_max_len: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization for misconception. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    max_example_num_per_dataset: int = field(
        default=100000000, metadata={"help": "the max number of examples for each dataset"}
    )

    query_instruction_for_retrieval: str = field(
        default="", metadata={"help": "query: "}
    )
    misc_instruction_for_retrieval: str = field(
        default="", metadata={"help": "misconception: "}
    )

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")

@dataclass
class RetrieverTrainingArguments(TrainingArguments):
    loss_type: str = field(default='only logits')
