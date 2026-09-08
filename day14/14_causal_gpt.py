import torch, torch.nn as nn

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=8000; hidden=768; n_layers=6; seq_len=512

class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden) # 6.1M Fig 2.10
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False) # Fig 4.3
    def forward(self, x):
        # Fig 4.5 causal mask - 이 1줄이 15%->95% 마법
        causal_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=device, dtype=torch.bool), diagonal=1)
        h = self.embed(x)
        for lyr in self.layers:
            h = lyr(h, src_mask=causal_mask) # 미래 가림
        return self.lm_head(h)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20):
        for _ in range(max_new_tokens):
            logits = self(idx[:, -seq_len:])[:, -1, :]
            next_idx = torch.argmax(logits, dim=-1, keepdim=True) # greedy Fig 4.16
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

model = CausalGPT().to(device=device, dtype=dtype)
print(f"Param: {sum(p.numel() for p in model.parameters())/1e6:.2f}M Fig 4.5 causal")

torch.manual_seed(0)
base_seq = torch.randint(0, vocab_size, (seq_len+1,), device=device)
x = base_seq[:-1].unsqueeze(0).repeat(2,1)
y = base_seq[1:].unsqueeze(0).repeat(2,1)

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
print("Training 100 steps with causal mask...")
for step in range(1,101):
    opt.zero_grad()
    loss = loss_fn(model(x).view(-1, vocab_size).float(), y.view(-1))
    loss.backward(); opt.step()
    if step==1 or step%25==0:
        print(f" {step:3d} | loss {loss.item():.4f} | CUDA {torch.cuda.memory_allocated()/1024**2:.0f}MB | Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

model.eval()
prompt_short = base_seq[:10].unsqueeze(0)
gen_short = model.generate(prompt_short, 20)
match_short = (gen_short[0][10:30]==base_seq[10:30]).float().mean()*100
print(f"\nShort 10->30 Match: {match_short:.1f}%")

prompt_long = base_seq[:492].unsqueeze(0)
gen_long = model.generate(prompt_long, 20)
match_long = (gen_long[0][492:512]==base_seq[492:512]).float().mean()*100
print(f"Long 492->512 Match: {match_long:.1f}% <- 80~100%면 성공 Fig 4.15")

print(f"Real: {base_seq[492:497].tolist()} | Gen: {gen_long[0][492:497].tolist()}")
