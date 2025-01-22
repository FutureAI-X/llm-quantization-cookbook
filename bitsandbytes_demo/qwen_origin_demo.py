from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "C:\Models\qwen2.5-0.5b-instruct"
model_name_save = "C:\Models\qwen2.5-0.5b-instruct-int4"

model = AutoModelForCausalLM.from_pretrained(
    model_name_save,
    torch_dtype="auto",
    device_map="auto"
)

memory_in_bytes = model.get_memory_footprint()
memory_in_mb = memory_in_bytes / 1024 / 1024
print(f"模型内存占用: {memory_in_mb:.2f} MB")

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "介绍一下浙江"
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)