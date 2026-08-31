import torch
import torch.nn as nn
import time

torch.manual_seed(42)
device = 'cuda'
dtype = torch.bfloat16

# Mini Transformer 0.59M from Dat2 -2 layer for speed
vocab, hidden, layers = 1000, 768, 2
model = nn.ModuleList([
    nn.Embedding(vocab, hidden),
    *[nn.TransformerEncoderLayer(hidden, 4, hidden*4, batch_first=True) for _ in range(layers)],
    nn.Linear(hidden, vocab)
]).to(device=device, dtype=dtype)

params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Param: {params:.2f}M - Day2 0.59M scaled down")

# Fake data from Day3 DataLoader batch 8
batch, seq = 8, 32
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

start = time.time()
print("Step | loss | time")
for step in range(1, 201):
    x = torch.randint(0, vocab, (batch, seq), device=device)
    y = torch.randint(0, vocab, (batch, seq), device=device)

    # Foward: emb -> trandformer -> logits
    h = model[0](x)
    for lyr in model[1:-1]:
        h = lyr(h)
    logits = model[-1](h)

    loss = nn.functional.cross_entropy(logits.view(-1, vocab).float(), y.view(-1))

    loss.backward()
    opt.step()
    opt.zero_grad()

    if step == 1 or step % 20 == 0:
        elapsed = time.time() - start
        print(f"{step:3d} | {loss.item():.4f} | {elapsed:.1f}s")

    if step == 200:
        print(f"Final: loss {loss.item():.4f} - target 0.0999")
        print(f"total: {time.time()-start:.1f}s - target 67s")
        print(f"CUDA: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
        print(f"GPU: {torch.cuda.get_device_name(0)}")


