import torch, tiktoken, torch.nn as nn
enc = tiktoken.get_encoding("cl100k_base")
vocab_size, hidden = 100277, 768
class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 12, hidden*4, batch_first=True) for _ in range(6)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
    def forward(self, input_ids):
        h = self.embed(input_ids)
        mask = torch.triu(torch.ones(h.size(1), h.size(1), device="cuda", dtype=torch.bool), diagonal=1)
        for lyr in self.layers: h = lyr(h, src_mask=mask)
        return self.lm_head(h)

model = CausalGPT().to("cuda").to(torch.bfloat16)
# Day18 LoRA 가중치 있으면 로드 (없어도 메모리 동일)
print(f"HF Serving initial: CUDA {torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
prompt="EN: Hello world, this is Day15 KO:"
ids=torch.tensor([enc.encode(prompt)], device="cuda")
with torch.no_grad():
    for i in range(10):
        logits=model(ids)
        nxt=torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)
        ids=torch.cat([ids,nxt], dim=1)
        if i%3==0:
            print(f"Generate {i} CUDA {torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
print("GEN HF Serving:", enc.decode(ids[0].tolist()))
print(f"Final HF Serving Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
