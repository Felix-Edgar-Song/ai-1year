import torch
import torch.nn as nn
import time

torch.manual_seed(42)
device = 'cuda'
dtype = torch.bfloat16

vocab_size = 500
hidden = 128  # 128 hidden for 0.59M target
n_layers = 4

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)  # token embedding
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*2, batch_first=True)
            for _ in range(n_layers)
        ])  # 4 layers transformer
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)  # output projection
    def forward(self, x):
        h = self.embed(x)  # [B,S] -> [B,S,H]
        for lyr in self.layers:
            h = lyr(h)  # transformer
        return self.lm_head(h)  # [B,S,V]

model = TinyGPT().to(device=device, dtype=dtype)
param_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Param: {param_m:.2f}M - target 0.59M")

# Fixed learnable dataset: y = x (copy task) to enable memorization
# If y is independent random, loss cannot go below ln(vocab)=6.21
torch.manual_seed(0)
fixed_data = []
for _ in range(20):  # 20 samples * repeat 10x = 200 steps
    x = torch.randint(0, vocab_size, (8, 32), device=device)
    y = x.clone()  # learnable mapping: predict input itself
    fixed_data.append((x, y))

opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.01)  # higher lr for fast memorization
loss_fn = nn.CrossEntropyLoss()

print("Step | loss | time | CUDA")
start = time.time()
for step in range(1, 201):
    x, y = fixed_data[step % len(fixed_data)]  # repeat 20 samples
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1))
    loss.backward()  # 0.66M grads
    opt.step()  # AdamW with m+v
    opt.zero_grad()

    if step == 1 or step % 20 == 0:
        print(f"{step:3d} | {loss.item():.4f} | {time.time()-start:.1f}s | {torch.cuda.memory_allocated()/1024**2:.1f}MB")

print(f"Final: loss {loss.item():.4f} - target 0.0999")
print(f"Total: {time.time()-start:.1f}s - target 67s")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA peak: {torch.cuda.max_memory_allocated()/1024**2:.1f}MB")
