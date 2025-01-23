from transformers import AutoModelForCausalLM, BitsAndBytesConfig

"""使用 bitsandbytes 对 Qwen 进行 Int8 量化"""
# Step1 定义原始模型路径 and 量化后保存路径
model_source_path = "C:\Models\qwen2.5-0.5b-instruct"
model_quantization_path = "C:\Models\qwen2.5-0.5b-instruct-bitsandbytes-int8"

# Step2 定义量化配置
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

# Step3 加载模型 并 执行量化
model = AutoModelForCausalLM.from_pretrained(
    model_source_path,
    device_map="auto",
    quantization_config=quantization_config
)

# Step4 保存量化模型
model.save_pretrained(model_quantization_path)