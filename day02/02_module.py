import torch
import torch.nn as nn

# 1. Llama 7B의 한 레이어 = hidden 4096이 아니라, 학습용 768로 축소해서 감각 익히기
hidden = 768
linear = nn.Linear(hidden, hidden, bias=True)

# 2. 파라미터 직접 세기 - 웹으로 치면 DB row count 세는 느낌
param_count = sum(p.numel() for p in linear.parameters())
# Linear = weight [768,768] + bias [768]
# 768*768 = 589,824 + 768 = 590,592 = 0.59M
print(f"nn.Linear({hidden},{hidden}) param = {param_count:,} = {param_count/1e6:.2f}M")
print(f" - weight: {linear.weight.numel():,} = {hidden}x{hidden}")
print(f" - bias: {linear.bias.numel():,} = {hidden}")
print()

# 3. 어제 QLoRA 540K와 비교
lora_rank = 8
lora_params = hidden * lora_rank * 2 # A [768,8] + B [8,768]
print(f"LoRA r={lora_rank} param = {lora_params:,} = {lora_params/1e3:.0f}K")
print(f"비율: {lora_params/param_count*100:.4f}% -> 어제 0.1093%랑 같은 원리!")

# 4. VRAM 연결 - 어제 bf16 50% 적용
mem_fp32 = param_count * 4 / 1024**2
mem_bf16 = param_count * 2 / 1024**2
print(f"\nVRAM: fp32 {mem_fp32:.2f}MB vs bf16 {mem_bf16:.2f}MB (50% saving, 어제 증명)")

# 5. CUDA에서 실제 할당 확인
linear_cuda = linear.to(dtype=torch.bfloat16, device='cuda')
print(f"\nCUDA allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
print(f"GPU: {torch.cuda.get_device_name(0)}")
