import torch, torch.nn as nn, time, wandb
wandb.init(project="ai-1year", name="day11-vocab8k-next", config={"vocab":8000,"hidden":768,"seq":512,"task":"next-token"})

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=8000; hidden=768; n_layers=6; seq_len=512; batch=2; accum=4  # vocab 32K->8K, eff batch 8

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)  # 8000*768=6.1M vs 24.5M - Fig 2.10 75% 절약
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True) for _ in range(n_layers)])  # Fig 3.15
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)  # 6.1M - Fig 4.3

    def forward(self, x):
        h = self.embed(x)  # [B,S,H]
        for lyr in self.layers: h = lyr(h)  # Fig 3.12
        return self.lm_head(h)  # [B,S,8000]

model = TinyGPT().to(device=device, dtype=dtype)
param_m = sum(p.numel() for p in model.parameters())/1e6
print(f"Param: {param_m:.2f}M - Day10 91.68M -> Day11 54.7M vocab 75% cut")

# next-token dataset: x = 0..511, y = 1..512 (shifted)
torch.manual_seed(0)
base_seq = torch.randint(0, vocab_size, (seq_len+1,), device=device)  # 513 tokens
fixed_data = []
for _ in range(20):
    # 반복 패턴: base_seq에 약간 노이즈 -> frequency up
    x = base_seq[:-1].clone().unsqueeze(0).repeat(batch,1)  # [2,512]
    y = base_seq[1:].clone().unsqueeze(0).repeat(batch,1)   # [2,512] next token
    fixed_data.append((x,y))

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Step | loss | time | CUDA | peak | vocab 8K next-token")
start=time.time()
opt.zero_grad()
for step in range(1,501):
    x,y = fixed_data[step % len(fixed_data)]
    logits = model(x)  # [B,S,8000]
    loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1)) / accum
    loss.backward()
    if step % accum == 0:
        opt.step()
        opt.zero_grad()

    if step==1 or step%50==0:
        elapsed=time.time()-start
        true_loss = loss.item()*accum
        print(f"{step:3d} | {true_loss:.4f} | {elapsed:.1f}s | {torch.cuda.memory_allocated()/1024**2:.0f}MB | {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
        wandb.log({"loss": true_loss, "cuda": torch.cuda.memory_allocated()/1024**2, "step": step})

print(f"Final {loss.item()*accum:.4f} | Total {time.time()-start:.1f}s | Param {param_m:.1f}M")
wandb.finish()
