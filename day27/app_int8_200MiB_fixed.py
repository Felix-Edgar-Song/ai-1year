import torch
import torch.nn as nn

# Blackwell 최적 설정 / Blackwell optimal config
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.triton.cudagraphs = False

class SimpleGPT(nn.Module):
    def __init__(self, vocab=50257, d_model=768, n_layer=12):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, 8, 768, batch_first=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
    def forward(self, x):
        h = self.wte(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.ln_f(h))

print("=== Day26 bf16 baseline 233MiB ===")
model_bf16 = SimpleGPT().to(dtype=torch.bfloat16, device='cuda')
model_bf16 = torch.compile(model_bf16, mode="reduce-overhead", fullgraph=False)
print(f"Before: {torch.cuda.memory_allocated()/1024**2:.1f}MiB")
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = torch.randint(0, 50257, (1, 8), device='cuda')
    for _ in range(2):
        out = model_bf16(x)
    torch.cuda.empty_cache()
    print(f"After 2x bf16: {torch.cuda.memory_allocated()/1024**2:.1f}MiB")
    print(f"Peak bf16: {torch.cuda.max_memory_allocated()/1024**2:.1f}MiB")
    torch.cuda.reset_peak_memory_stats()

print("\n=== Day27 int8 weight-only 200MiB challenge ===")
# torchao int8_weight_only - CUDA에서 동작하는 유일한 int8 / Only int8 that works on CUDA
from torchao.quantization import quantize_, int8_weight_only

model_int8 = SimpleGPT().to(device='cuda') # fp32로 생성 후 int8 변환
quantize_(model_int8, int8_weight_only()) # int8 weight only - CUDA 지원
model_int8 = model_int8.to(dtype=torch.bfloat16) # 나머지 bf16 유지
model_int8 = torch.compile(model_int8, mode="reduce-overhead", fullgraph=False)

print(f"Before int8: {torch.cuda.memory_allocated()/1024**2:.1f}MiB")
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = torch.randint(0, 50257, (1, 8), device='cuda')
    for _ in range(2):
        out = model_int8(x)
    torch.cuda.empty_cache()
    print(f"After 2x int8: {torch.cuda.memory_allocated()/1024**2:.1f}MiB / 실행 후")
    print(f"Peak int8: {torch.cuda.max_memory_allocated()/1024**2:.1f}MiB / 최대")
    print(f"Output shape: {out.shape}")
