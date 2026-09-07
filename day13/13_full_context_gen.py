import torch, torch.nn as nn, time

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=8000; hidden=768; n_layers=6; seq_len=512

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden) # 6.1M Fig 2.10
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True) for _ in range(n_layers)]) # Fig 3.15
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False) # Fig 4.3
    def forward(self, x):
        h = self.embed(x)
        for lyr in self.layers: h = lyr(h)
        return self.lm_head(h)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20, temperature=0.1):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -seq_len:])[:, -1, :] / temperature
            probs = torch.softmax(logits.float(), dim=-1)
            next_idx = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

model = TinyGPT().to(device=device, dtype=dtype)
print(f"Param: {sum(p.numel() for p in model.parameters())/1e6:.2f}M Fig 4.3")

torch.manual_seed(0)
base_seq = torch.randint(0, vocab_size, (seq_len+1,), device=device) # 513
x = base_seq[:-1].unsqueeze(0).repeat(2,1)
y = base_seq[1:].unsqueeze(0).repeat(2,1)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
print("Training 100 steps...")
for step in range(1,101):
    opt.zero_grad()
    loss = loss_fn(model(x).view(-1, vocab_size).float(), y.view(-1))
    loss.backward(); opt.step()
    if step%25==0:
        print(f" {step:3d} | {loss.item():.4f} | {torch.cuda.memory_allocated()/1024**2:.0f}MB")

# checkpoint 저장 Fig 4.17
torch.save(model.state_dict(), "ckpt_54M_0.04.pt")
print(f"Saved ckpt_54M_0.04.pt {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# 테스트 1: 짧은 prompt 10개 -> 40% (어제 복습)
prompt_short = base_seq[:10].unsqueeze(0)
gen_short = model.generate(prompt_short, 20, temperature=0.1)
match_short = (gen_short[0][10:20]==base_seq[10:20]).float().mean()*100
print(f"\nShort prompt 10->20 Match: {match_short:.1f}% (expected 40% Fig 4.15 exposure bias)")

# 테스트 2: 긴 prompt 492개 -> 95% (오늘 목표)
prompt_long = base_seq[:492].unsqueeze(0)
gen_long = model.generate(prompt_long, 20, temperature=0.1)
match_long = (gen_long[0][492:512]==base_seq[492:512]).float().mean()*100
print(f"Long prompt 492->512 Match: {match_long:.1f}% (expected 90~100% full context)")

print(f"\nSeed 10: {prompt_short[0].tolist()[:5]}... | Gen short 10: {gen_short[0].tolist()[10:15]}... | Real {base_seq[10:15].tolist()}")
print(f"Seed 492: {prompt_long[0].tolist()[-5:]}... | Gen long 20: {gen_long[0].tolist()[492:497]}... | Real {base_seq[492:497].tolist()}")
