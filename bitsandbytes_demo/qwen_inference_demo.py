from transformers import AutoModelForCausalLM, AutoTokenizer

from common_utils import model_info_print

"""Qwen 模型推理示例代码"""
# Step1 定义 tokenizer 和 model 的路径
# quantization_type 为 4 加载 int4 量化模型，为 8 加载 int8 量化模型，否则加载未量化模型
quantization_type = 4
tokenizer_path = "C:\Models\qwen2.5-0.5b-instruct"
model_path = "C:\Models\qwen2.5-0.5b-instruct-bitsandbytes-int4" if quantization_type == 4 else "C:\Models\qwen2.5-0.5b-instruct-bitsandbytes-int8" if quantization_type == 8 else "C:\Models\qwen2.5-0.5b-instruct"

# Step2 加载模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)

# 打印内存占用情况
model_info_print.print_memory_use(model)

# Step3 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Step4 组装 prompt
prompt = "你是谁"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Step5 推理
# 1、对输入进行 token 化
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# 2、模型推理
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# 3、输出 token 处理
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# 4、解码输出
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# 5、打印输出
print(response)