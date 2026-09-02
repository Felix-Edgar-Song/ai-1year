import torch, torch.nn as nn, time, wandb
wandb.init(project="ai-1year", name="day08-seq512-67s", config={"hidden":768,"layers":6,"seq":512,"batch":4})

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=500; hidden=768; n_layers=6; seq_len=512; batch=4  # seq 128->512 4x, batch 8->4 for 16GB

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)  # 500*768=384K
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True)
            for _ in range(n_layers)
        ])  # 6 layers attention O(n^2)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)  # 768->500

    def forward(self, x):
        h = self.embed(x)  # [B,S,H] B=4 S=512
        for lyr in self.layers:
            h = lyr(h)  # attention 512*512=262K ops per layer
        return self.lm_head(h)

model = TinyGPT().to(device=device, dtype=dtype)
print(f"Param: {sum(p.numel() for p in model.parameters())/1e6:.2f}M - same as Day7 43M")

# Fixed dataset y=x copy task for memorization
torch.manual_seed(0)
fixed_data = [(torch.randint(0, vocab_size, (batch, seq_len), device=device), None) for _ in range(20)]
fixed_data = [(x, x.clone()) for x,_ in fixed_data]

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)  # AdamW
loss_fn = nn.CrossEntropyLoss()

print("Step | loss | time | CUDA | peak | seq=512 O(n^2)")
start=time.time()
for step in range(1,201):
    x,y = fixed_data[step % len(fixed_data)]
    t0=time.time()
    logits = model(x)
    fwd = time.time()-t0
    loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()

    if step==1 or step%20==0:
        elapsed=time.time()-start
        cuda_mb=torch.cuda.memory_allocated()/1024**2
        peak_mb=torch.cuda.max_memory_allocated()/1024**2
        print(f"{step:3d} | {loss.item():.4f} | {elapsed:.1f}s | {cuda_mb:.0f}MB | {peak_mb:.0f}MB | fwd {fwd*1000:.0f}ms")
        wandb.log({"loss": loss.item(), "cuda": cuda_mb, "peak": peak_mb, "fwd_ms": fwd*1000, "step": step, "elapsed": elapsed})

print(f"Final loss {loss.item():.4f} Target 0.0999 | Total {time.time()-start:.1f}s Target 67s | GPU {torch.cuda.get_device_name(0)}")
wandb.finish()
