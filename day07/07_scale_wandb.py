import torch
import torch.nn as nn
import time
import wandb

# Init wandb - web dashboard for training
wandb.init(project="ai-1year", name="day07-768-hidden-67s", config={
    "vocab": 500, "hidden": 768, "layers": 6, "seq": 128, "batch": 8
})

torch.manual_seed(42)
device = 'cuda'
dtype = torch.bfloat16

vocab_size = 500
hidden = 768  # scaled up from 128 to 768 for 67s target
n_layers = 6
seq_len = 128
batch = 8

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)  # 500*768=384K
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True)
            for _ in range(n_layers)
        ])  # 6 layers, ~4.2M
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)  # 768*500=384K
    def forward(self, x):
        h = self.embed(x)  # [B,S,H]
        for lyr in self.layers:
            h = lyr(h)
        return self.lm_head(h)

model = TinyGPT().to(device=device, dtype=dtype)
param_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Param: {param_m:.2f}M - scaled from 0.66M to 4.7M")

# Fixed dataset 20 samples copy task
torch.manual_seed(0)
fixed_data = [(torch.randint(0, vocab_size, (batch, seq_len), device=device),
               torch.randint(0, vocab_size, (batch, seq_len), device=device)) for _ in range(20)]
fixed_data = [(x, x.clone()) for x, _ in fixed_data]  # y=x copy task

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)  # lower lr for larger model
loss_fn = nn.CrossEntropyLoss()

print("Step | loss | time | CUDA | peak")
start = time.time()
for step in range(1, 201):
    x, y = fixed_data[step % len(fixed_data)]
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()

    if step % 10 == 0:
        elapsed = time.time() - start
        cuda_mb = torch.cuda.memory_allocated() / 1024**2
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        print(f"{step:3d} | {loss.item():.4f} | {elapsed:.1f}s | {cuda_mb:.1f}MB | {peak_mb:.1f}MB")
        wandb.log({"loss": loss.item(), "cuda_mb": cuda_mb, "step": step, "elapsed": elapsed})

    if step == 1:
        print(f"{step:3d} | {loss.item():.4f} | {time.time()-start:.1f}s | {torch.cuda.memory_allocated()/1024**2:.1f}MB | {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")

elapsed = time.time() - start
print(f"Final: loss {loss.item():.4f} Target 0.0999")
print(f"Total: {elapsed:.1f}s Target 67s")
print(f"GPU: {torch.cuda.get_device_name(0)}")
wandb.finish()
