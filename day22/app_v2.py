from pathlib import Path
import os, torch, torch.nn as nn, tiktoken
from fastapi import FastAPI
import torch.nn.functional as F
from asyncio import Semaphore

CKPT = Path(os.getenv("CKPT", "/home/felix/ai-1year/day18/sft_1000.pt"))
app = FastAPI()
sem = Semaphore(10) # 동시 10개 제한 - 2274MiB 폭주 방지
VOCAB, HIDDEN, LAYERS, HEADS = 100277, 768, 6, 12

class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.pos_embed = nn.Embedding(512, HIDDEN)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(HIDDEN, HEADS, HIDDEN*4, batch_first=True, norm_first=True) for _ in range(LAYERS)])
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    def forward(self, x):
        B,T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for l in self.layers: h = l(h)
        return self.lm_head(self.ln_f(h))

@app.on_event("startup")
def load():
    global model, enc
    assert CKPT.exists()
    model = CausalGPT().to("cuda").to(torch.bfloat16)
    ckpt = torch.load(CKPT, map_location="cuda")
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Model 757MB loaded")

@app.get("/generate")
async def generate(prompt: str = "EN: Hello KO:", max_new: int = 12, temp: float = 0.0):
    torch.cuda.reset_peak_memory_stats()
    async with sem: # 10개 제한
        ids = torch.tensor([enc.encode(prompt)], device="cuda")
        with torch.no_grad():
            for _ in range(max_new):
                logits = model(ids)[:, -1, :]
                if temp == 0:
                    nxt = logits.argmax(-1, keepdim=True)
                else:
                    nxt = torch.multinomial(F.softmax(logits/0.8, -1), 1)
                ids = torch.cat([ids, nxt], dim=1)
        gen = enc.decode(ids[0].tolist())
    vram = torch.cuda.memory_allocated()/1024**2
    del ids; torch.cuda.empty_cache()
    return {"gen": gen, "vram": f"{vram:.0f}MB"}

@app.get("/health")
def health():
    return {"vram": torch.cuda.memory_allocated()/1024**2}
