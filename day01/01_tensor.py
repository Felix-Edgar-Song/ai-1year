import torch
print(f"CUDA:{torch.cuda.is_available()} GPU:{torch.cuda.get_device_name(0)} {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
# bf16 50% 증명
a = torch.randn(4096,4096, dtype=torch.float32, device='cuda')
mem_fp32 = torch.cuda.memory_allocated()/1024**2
del a; torch.cuda.empty_cache()
b = torch.randn(4096,4096, dtype=torch.bfloat16, device='cuda')
mem_bf16 = torch.cuda.memory_allocated()/1024**2
print(f"fp32 {mem_fp32:.1f}MB vs bf16 {mem_bf16:.1f}MB -> {mem_bf16/mem_fp32*100:.0f}% = 50% saving")
