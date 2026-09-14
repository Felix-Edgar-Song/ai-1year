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
ckpt = torch.load("/home/felix/ai-1year/day18/sft_1000.pt", map_location="cuda")
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
enc = tiktoken.get_encoding("cl100k_base")
print(f"Model loaded 375M - {torch.cuda.memory_allocated()/1024**2:.0f}MB")

@app.get("/generate")
def generate(prompt: str = "EN: Hello KO:"):
    ids = torch.tensor([enc.encode(prompt)], device="cuda")
    for _ in range(12):
        with torch.no_grad(): nxt = model(ids)[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return {"gen": enc.decode(ids[0].tolist()), "vram": f"{torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB"}
