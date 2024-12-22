import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, PeftModel

def count_parameters(model):
    """
    打印参数的数量
    """
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
    print(f"base model params: {all_param} || ")


def get_model(model_args, training_args):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    loraConfig = LoraConfig(
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        target_modules=model_args.target_modules.split(","),
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
        inference_mode=False,  # 确保不是推理模式
    )

    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    ### base_model for encoder
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, 
        # torch_dtype=torch.bfloat16,
        use_flash_attention_2=True if model_args.use_flash_attn else False,
        token=model_args.token,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True, 
        quantization_config=bnb_config)

    count_parameters(model)

    # 冻结基础模型参数
    for param in model.parameters():
        param.requires_grad = False

    model.config.use_cache = False #这句干嘛的？ 不懂

    model = get_peft_model(model, loraConfig)

    if model_args.lora_path !='none':
        print("加载之前的lora参数...", model_args.lora_path)
        d = torch.load(model_args.lora_path, map_location=model.device)
        model.load_state_dict(d, strict=False)

        
    # $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    model.print_trainable_parameters()

    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # added for stage 3不报错
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.bfloat16)  # 确保所有 trainable 参数 dtype 一致
    # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    return model