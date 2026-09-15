from pathlib import Path
import os, json, torch, torch.nn as nn, tiktoken
from fastapi import FastAPI
import torch.nn.functional as F

CKPT = Path(os.getenv("CKPT", Path(__file__).parent.parent / "day18/sft_1000.pt"))
app = FastAPI()

# Day18 375M config - Day20 실측 756MB
VOCAB = 100277
HIDDEN = 768
LAYERS = 6
HEADS = 12

class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.pos_embed = nn.Embedding(512, HIDDEN)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=HIDDEN, nhead=HEADS, dim_feedforward=HIDDEN*4,
                batch_first=True, norm_first=True
            ) for _ in range(LAYERS)
        ])
        self.ln_f = nn.LayerNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.lm_head(h)

@app.on_event("startup")
def load():
    global model, enc
    assert CKPT.exists(), f"CKPT not found: {CKPT} - run day18/sft_1000.pt first"
    print(f"Loading CKPT: {CKPT} exists={CKPT.exists()}")
    model = CausalGPT().to("cuda").to(torch.bfloat16)
    ckpt = torch.load(CKPT, map_location="cuda")
    # ckpt가 dict일 수도, state_dict일 수도
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False) # Day20처럼 strict=False로 복구
    model.eval()
    global enc
    enc = tiktoken.get_encoding("cl100k_base")
    print(f"Model loaded 375M - {torch.cuda.memory_allocated()/1024**2:.0f}MB")

@app.get("/health")
def health():
    return {"ckpt": str(CKPT), "exists": CKPT.exists(), "vram_MB": torch.cuda.memory_allocated()/1024**2}

@app.get("/generate")
def generate(prompt: str = "EN: Hello KO:", max_new: int = 12, temp: float = 0.8):
    torch.cuda.reset_peak_memory_stats()
    ids = torch.tensor([enc.encode(prompt)], device="cuda")
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(ids)[:, -1, :] / temp
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)
    gen = enc.decode(ids[0].tolist())
    vram = torch.cuda.memory_allocated()/1024**2
    peak = torch.cuda.max_memory_allocated()/1024**2
    del ids
    torch.cuda.empty_cache()
    return {"gen": gen, "vram": f"{vram:.0f}MB Peak {peak:.0f}MB"}

@app.get("/")
def root():
    return {"msg": "Day21 1100MiB FastAPI", "ckpt": str(CKPT)}
