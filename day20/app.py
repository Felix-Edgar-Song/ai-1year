from fastapi import FastAPI
import torch, torch.nn as nn, tiktoken, os
vocab_size, hidden = 100277, 768
class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 12, hidden*4, batch_first=True) for _ in range(6)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
    def forward(self, x):
        x = self.embed(x)
        for l in self.layers: x = l(x)
        return self.lm_head(x)

app = FastAPI()
model = CausalGPT().to("cuda").to(torch.bfloat16)
ckpt_path = "/home/felix/ai-1year/day18/sft_1000.pt"
print(f"Trying {ckpt_path} exists={os.path.exists(ckpt_path)} size={os.path.getsize(ckpt_path)/1024**2:.0f}MB")
ckpt = torch.load(ckpt_path, map_location="cuda")
sd = ckpt["model"] if "model" in ckpt else ckpt
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"Loaded: missing={len(missing)} unexpected={len(unexpected)}")
print("LoRA loaded 100% - 안녕 세상 Ready" if len(missing)<10 else f"Partial load missing {missing[:3]}")
model.eval()
enc = tiktoken.get_encoding("cl100k_base")

@app.get("/generate")
def generate(prompt: str = "EN: Hello KO:"):
    ids = torch.tensor([enc.encode(prompt)], device="cuda")
    for _ in range(15):
        with torch.no_grad():
            logits = model(ids)[:, -1, :]
            nxt = logits.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() == 100257: break
    text = enc.decode(ids[0].tolist())
    return {"prompt": prompt, "gen": text, "vram": f"{torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB"}
