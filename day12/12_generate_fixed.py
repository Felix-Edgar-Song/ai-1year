import torch, torch.nn as nn, time

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=8000; hidden=768; n_layers=6; seq_len=512

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden) # 6.1M Fig 2.10
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True) for _ in range(n_layers)]) # Fig 3.15
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False) # 6.1M Fig 4.3
    def forward(self, x):
        h = self.embed(x)
        for lyr in self.layers: h = lyr(h) # Fig 3.12 O(n^2)
        return self.lm_head(h)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20, temperature=0.8): # Fig 4.15
        for _ in range(max_new_tokens):
            logits = self(idx[:, -seq_len:])[:, -1, :] / temperature
            probs = torch.softmax(logits.float(), dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1) # Fig 4.16 sampling
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

model = TinyGPT().to(device=device, dtype=dtype)
print(f"Param: {sum(p.numel() for p in model.parameters())/1e6:.2f}M - same 54.82M Fig 4.3")

# Day11 base_seq 고정
torch.manual_seed(0)
base_seq = torch.randint(0, vocab_size, (seq_len+1,), device=device) # 513 tokens - Fig 2.5
x = base_seq[:-1].unsqueeze(0).repeat(2,1) # [2,512]
y = base_seq[1:].unsqueeze(0).repeat(2,1) # [2,512] next-token Fig 5.2

# 100 steps 재학습 - Day11 0.04 재현
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
print("Training 100 steps to 0.04...")
for step in range(1,101):
    opt.zero_grad()
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1))
    loss.backward(); opt.step()
    if step==1 or step%25==0:
        print(f" {step:3d} | loss {loss.item():.4f} | CUDA {torch.cuda.memory_allocated()/1024**2:.0f}MB | Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

# 생성 테스트 - 학습된 모델로
model.eval()
prompt = base_seq[:10].unsqueeze(0) # 10 tokens seed Fig 4.15
generated = model.generate(prompt, max_new_tokens=20, temperature=0.1) # temp 0.1 greedy

print(f"\nSeed 10: {prompt[0].tolist()}")
print(f"Generated 20: {generated[0].tolist()[10:]}")
print(f"Real 20: {base_seq[10:30].tolist()}")
match = (generated[0][10:30]==base_seq[10:30]).float().mean()*100
print(f"Match: {match:.1f}% - loss {loss.item():.4f} => {100*2.718**(-loss.item()):.1f}% expected (Fig 4.15)")

# Next-token check
with torch.no_grad():
    pred = model(prompt).argmax(dim=-1)
    print(f"\nNext-token check: pred {pred[0,-1].item()} vs true {base_seq[10].item()} - should match if loss 0.04")

# temperature 비교 Fig 4.16
for temp in [0.1, 0.8, 1.5]:
    gen = model.generate(prompt, max_new_tokens=10, temperature=temp)
    print(f"temp {temp}: {gen[0].tolist()[10:20]}")

