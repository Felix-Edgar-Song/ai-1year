import torch, torch.nn as nn, time, wandb
wandb.init(project="ai-1year", name="day09-vocab32k-67m", config={"vocab":32000,"hidden":768,"seq":512})

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=32000; hidden=768; n_layers=6; seq_len=512; batch=2  # vocab 64x, batch 4->2 for 16GB

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)  # 32000*768=24.5M
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True) for _ in range(n_layers)])  # 43M
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)  # 24.5M

    def forward(self, x):
        h = self.embed(x)  # [B,S,H]
        for lyr in self.layers: h = lyr(h)  # 512*512 attention
        return self.lm_head(h)  # [B,S,32000]

model = TinyGPT().to(device=device, dtype=dtype)
param_m = sum(p.numel() for p in model.parameters())/1e6
print(f"Param: {param_m:.2f}M - Day8 43M -> Day9 91M vocab 64x")

torch.manual_seed(0)
fixed_data = [(torch.randint(0, vocab_size, (batch, seq_len), device=device), None) for _ in range(20)]
fixed_data = [(x, x.clone()) for x,_ in fixed_data]  # y=x copy task

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
loss_fn = nn.CrossEntropyLoss()

print("Step | loss | time | CUDA | peak | vocab 32K")
start=time.time()
for step in range(1,201):
    x,y = fixed_data[step % len(fixed_data)]
    loss = loss_fn(model(x).view(-1, vocab_size).float(), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    if step==1 or step%20==0:
        elapsed=time.time()-start
        print(f"{step:3d} | {loss.item():.4f} | {elapsed:.1f}s | {torch.cuda.memory_allocated()/1024**2:.0f}MB | {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
        wandb.log({"loss": loss.item(), "cuda": torch.cuda.memory_allocated()/1024**2, "step": step})

print(f"Final {loss.item():.4f} | Total {time.time()-start:.1f}s | Param {param_m:.1f}M | GPU {torch.cuda.get_device_name(0)}")
wandb.finish()
