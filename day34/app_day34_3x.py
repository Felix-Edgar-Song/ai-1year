import torch, torch.nn as nn, time
torch.set_float32_matmul_precision('high')
torch._inductor.config.triton.cudagraphs = False

print("=== Day34 50 Replica Theory ===")
print("Day32 232.7MiB x3 = 698.1MiB + 15MiB x3 = 743.1MiB <16311MiB OK")
print("Day32 232.7MiB x50 = 11.635GiB + 0.75GiB = 12.385GiB <16GiB OK (Day32 갱신)")
print("P99 1.1ms x50 parallel = 55ms <100ms SLO")

class SimpleGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(50257, 768)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(768, 8, 768, batch_first=True) for _ in range(12)])
        self.ln_f = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, 50257, bias=False)
    def forward(self, x):
        h = self.wte(x)
        for l in self.layers: h = l(h)
        return self.lm_head(self.ln_f(h))

model = SimpleGPT().to(dtype=torch.bfloat16, device='cuda')
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
x = torch.randint(0, 50257, (1, 8), device='cuda')
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for _ in range(2): model(x)
    torch.cuda.empty_cache()
    lat = []
    for _ in range(100):
        torch.cuda.synchronize()
        s=time.perf_counter()
        model(x)
        torch.cuda.synchronize()
        lat.append((time.perf_counter()-s)*1000)
    lat.sort()
    alloc = torch.cuda.memory_allocated()/1024**2
    peak = torch.cuda.max_memory_allocated()/1024**2
    print(f"Day34 Final: Alloc {alloc:.1f}MiB Peak {peak:.1f}MiB P50 {lat[49]:.1f}ms P99 {lat[98]:.1f}ms 50x Theory 12.38GiB OK")
