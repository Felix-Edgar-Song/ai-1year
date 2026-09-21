import torch
import torch.nn as nn
from torch.ao.quantization import quantize_dynamic

torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

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

# bf16 233MiB 모델 로드
model = SimpleGPT().to(dtype=torch.bfloat16, device='cuda')
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
print(f"[bf16] Alloc: {torch.cuda.memory_allocated()/1024**2:.1f}MiB")

# int8 dynamic quantization - 200MiB 도전
model_int8 = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
print(f"[int8] Alloc: {torch.cuda.memory_allocated()/1024**2:.1f}MiB")

with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = torch.randint(0, 50257, (1, 8), device='cuda')
    for _ in range(2):
        out = model_int8(x)
    torch.cuda.empty_cache()
    print(f"After 2x: {torch.cuda.memory_allocated()/1024**2:.1f}MiB")
    print(f"Peak: {torch.cuda.max_memory_allocated()/1024**2:.1f}MiB")
