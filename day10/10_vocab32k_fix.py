import torch, torch.nn as nn, time, wandb
wandb.init(project="ai-1year", name="day10-vocab32k-fix", config={"vocab":32000,"hidden":768,"lr":1e-3,"grad_accum":4})

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=32000; hidden=768; n_layers=6; seq_len=512; batch=2; accum=4  # effective batch 8

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)  # 24.5M Figure 2.10
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True) for _ in range(n_layers)])  # Figure 3.15
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)  # 24.5M Figure 4.3

    def forward(self, x):
        h = self.embed(x)  # [B,S,H] B=2
        for lyr in self.layers: h = lyr(h)  # Figure 3.12 O(n^2)
        return self.lm_head(h)  # [B,S,32000]

model = TinyGPT().to(device=device, dtype=dtype)
param_m = sum(p.numel() for p in model.parameters())/1e6
print(f"Param: {param_m:.2f}M - Day9 91.68M same")

torch.manual_seed(0)
fixed_data = [(torch.randint(0, vocab_size, (batch, seq_len), device=device), None) for _ in range(20)]
fixed_data = [(x, x.clone()) for x,_ in fixed_data]  # y=x copy

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # lr 3e-4->1e-3 Figure 5.8
loss_fn = nn.CrossEntropyLoss()

print("Step | loss | time | CUDA | peak | accum 4")
start=time.time()
opt.zero_grad()
for step in range(1,501):  # 200->500
    x,y = fixed_data[step % len(fixed_data)]
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1)) / accum
    loss.backward()
    if step % accum == 0:
        opt.step()
        opt.zero_grad()

    if step==1 or step%50==0:
        elapsed=time.time()-start
        cuda_mb=torch.cuda.memory_allocated()/1024**2
        peak_mb=torch.cuda.max_memory_allocated()/1024**2
        true_loss = loss.item()*accum
        print(f"{step:3d} | {true_loss:.4f} | {elapsed:.1f}s | {cuda_mb:.0f}MB | {peak_mb:.0f}MB | eff batch 8")
        wandb.log({"loss": true_loss, "cuda": cuda_mb, "peak": peak_mb, "step": step})

print(f"Final {loss.item()*accum:.4f} | Total {time.time()-start:.1f}s | Param {param_m:.1f}M")
wandb.finish()
