import time, torch, torch.nn as nn
from fastapi import FastAPI
import uvicorn
from concurrent.futures import ThreadPoolExecutor
import statistics

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
        for l in self.layers: h = l(h)
        return self.lm_head(self.ln_f(h))

print("Loading 233MiB model...")
model = SimpleGPT().to(dtype=torch.bfloat16, device='cuda')
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
x = torch.randint(0, 50257, (1, 8), device='cuda')
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for _ in range(2): model(x)
torch.cuda.empty_cache()
print(f"Alloc {torch.cuda.memory_allocated()/1024**2:.1f}MiB / Peak {torch.cuda.max_memory_allocated()/1024**2:.1f}MiB")

# P99 측정
latencies = []
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    for i in range(100):
        start = time.perf_counter()
        _ = model(x)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter()-start)*1000)

latencies.sort()
p50 = latencies[49]
p99 = latencies[98]
print(f"\n=== P99 Result (100 runs) ===")
print(f"P50: {p50:.1f}ms / P99: {p99:.1f}ms / Mean: {statistics.mean(latencies):.1f}ms")
print(f"README 1줄: P50 {p50:.0f}ms P99 {p99:.0f}ms @ 233.4MiB 1 replica RTX 5060 Ti")
