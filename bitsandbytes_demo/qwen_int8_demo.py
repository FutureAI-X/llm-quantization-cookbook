from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

model_name = "C:\Models\qwen2.5-0.5b-instruct"
model_name_save = "C:\Models\qwen2.5-0.5b-instruct-int4"

quantization_config = BitsAndBytesConfig(load_in_4bit=True)

model_8bit = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)

model_8bit.save_pretrained(model_name_save)

memory_in_bytes = model_8bit.get_memory_footprint()
memory_in_mb = memory_in_bytes / 1024 / 1024
print(f"模型内存占用: {memory_in_mb:.2f} MB")
"""
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
model_inputs = tokenizer([text], return_tensors="pt").to(model_8bit.device)

generated_ids = model_8bit.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)
"""