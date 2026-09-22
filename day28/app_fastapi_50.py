from fastapi import FastAPI
import torch, torch.nn as nn

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

app = FastAPI()
model = SimpleGPT().to(dtype=torch.bfloat16, device='cuda')
model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

@app.get("/vram")
def vram():
    return {"allocated_MiB": torch.cuda.memory_allocated()/1024**2, "peak_MiB": torch.cuda.max_memory_allocated()/1024**2}

# 50개 포트: 8000-8049
print("Ready: 233MiB x 50 = 11.6GiB < 16GiB - uvicorn app_fastapi_50:app --port 8000")
