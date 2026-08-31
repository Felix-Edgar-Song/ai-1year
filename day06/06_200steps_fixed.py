import torch
import torch.nn as nn
import time

torch.manual_seed(42)
device = 'cuda'
dtype = torch.bfloat16

vocab_size = 500
hidden = 128  # scaled down from 768 to match 0.59M target
n_layers = 4

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)  # token embedding 500*128
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden*2, batch_first=True)
            for _ in range(n_layers)
        ])  # 4 transformer layers, ~0.5M params
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)  # output projection 128->500
    def forward(self, x):
        h = self.embed(x)  # [8,32] -> [8,32,128]
        for lyr in self.layers:
            h = lyr(h)  # transformer forward
        return self.lm_head(h)  # [8,32,500] logits

model = TinyGPT().to(device=device, dtype=dtype)
param_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Param: {param_m:.2f}M - target 0.59M (128 hidden)")

# Fixed dataset 100 samples - must be fixed to memorize, not random each step
torch.manual_seed(0)
fixed_data = [(torch.randint(0, vocab_size, (8, 32)), torch.randint(0, vocab_size, (8, 32))) for _ in range(100)]
fixed_data = [(x.to(device), y.to(device)) for x, y in fixed_data]

opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)  # AdamW adaptive lr from Day5
loss_fn = nn.CrossEntropyLoss()  # CE loss ln(vocab) ~6.2 start

print("Step | loss | time | CUDA")
start = time.time()
for step in range(1, 201):
    x, y = fixed_data[step % 100]  # repeat 100 samples 2x to memorize
    logits = model(x)
    loss = loss_fn(logits.view(-1, vocab_size).float(), y.view(-1))
    loss.backward()  # backward 0.59M grads, Day4 concept
    opt.step()  # AdamW step with m+v states
    opt.zero_grad()  # clear grads for next step

    if step == 1 or step % 20 == 0:
        print(f"{step:3d} | {loss.item():.4f} | {time.time()-start:.1f}s | {torch.cuda.memory_allocated()/1024**2:.1f}MB")

print(f"Final: loss {loss.item():.4f} - target 0.0999")
print(f"Total: {time.time()-start:.1f}s - target 67s")
print(f"GPU: {torch.cuda.get_device_name(0)}")
