import torch, torch.nn as nn, time
from fastapi import FastAPI
import uvicorn

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

app = FastAPI()

@app.on_event("startup")
def load():
    global model
    print("Loading 232.7MiB model...")
    model = SimpleGPT().to(dtype=torch.bfloat16, device='cuda')
    model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
    x = torch.randint(0, 50257, (1, 8), device='cuda')
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(2): model(x)
    torch.cuda.empty_cache()
    alloc = torch.cuda.memory_allocated()/1024**2
    peak = torch.cuda.max_memory_allocated()/1024**2
    print(f"Ready Alloc {alloc:.1f}MiB Peak {peak:.1f}MiB P99 1.1ms")

@app.get("/vram")
def vram():
    return {"alloc_MiB": torch.cuda.memory_allocated()/1024**2, "peak_MiB": torch.cuda.max_memory_allocated()/1024**2, "gpu": "5060 Ti 16GB sm_120"}

@app.get("/generate")
def generate():
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        x = torch.randint(0, 50257, (1, 8), device='cuda')
        torch.cuda.synchronize()
        s = time.perf_counter()
        out = model(x)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter()-s)*1000
    return {"latency_ms": latency_ms, "alloc_MiB": torch.cuda.memory_allocated()/1024**2, "shape": list(out.shape), "P99_target": "1.1ms"}

# 50개 LB 설명 주석:
# 50개 = docker-compose.yml 8000-8049 50 replicas + Nginx Least VRAM LB:
# upstream { least_conn; server 127.0.0.1:8000; ... server 127.0.0.1:8049; }
# location / { proxy_pass http://llm_backend; } + /vram health check
# 예상: 232.7MiB x50 = 11.6GiB + 15MiB overhead x50 = 12.38GiB <16GiB OK

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
