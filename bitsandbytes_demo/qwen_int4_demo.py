import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

"""使用 bitsandbytes 对 Qwen 进行 Int4 量化"""
# Step1 定义原始模型路径 and 量化后保存路径
model_name = "C:\Models\qwen2.5-0.5b-instruct"
model_name_save = "C:\Models\qwen2.5-0.5b-instruct-bitsandbytes-int4"

# Step2 定义量化配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

# Step3 加载模型 并 执行量化
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

# Step4 保存量化模型
model.save_pretrained(model_name_save)
