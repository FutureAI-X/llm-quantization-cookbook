"""模型信息打印类"""
def print_memory_use(model):
    memory_in_bytes = model.get_memory_footprint()
    memory_in_mb = memory_in_bytes / 1024 / 1024
    print(f"模型内存占用: {memory_in_mb:.2f} MB")