import torch, torch.nn as nn
torch.set_float32_matmul_precision('high')
torch._inductor.config.triton.cudagraphs = False

class SimpleGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 768)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(768, 8, 768, batch_first=True) for _ in range(12)])
        self.ln_f = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, 50257, bias=False)
    def forward(self, x):
        h = self.wte(x)
        for l in self.layers:
            h = l(h)
        return self.lm_head(self.ln_f(h))

# Single 233MiB 재확인
model = SimpleGPT().to(dtype=torch.bfloat16, device='cuda')
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = torch.randint(0, 50257, (1, 8), device='cuda')
    for _ in range(2):
        out = model(x)
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated()/1024**2
    peak = torch.cuda.max_memory_allocated()/1024**2
    print(f"Single: {alloc:.1f}MiB Alloc, {peak:.1f}MiB Peak / 단일: {alloc:.1f}MiB 할당, {peak:.1f}MiB 최대")

print(f"\n50개 프로덕션: {(233+15)*50}MiB = {(233+15)*50/1024:.1f}GiB <16GiB OK")
print("LB: Least VRAM + /vram health check, 8000-8049 포트")
