import torch, torch.nn as nn

torch.manual_seed(42)
device='cuda'; dtype=torch.bfloat16
vocab_size=8000; hidden=768; n_layers=6; seq_len=512

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden) # 6.1M Fig 2.10
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=hidden, nhead=8, dim_feedforward=hidden*4, batch_first=True) for _ in range(n_layers)]) # Fig 3.15
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False) # 6.1M Fig 4.3
    def forward(self, x):
        h = self.embed(x)
        for lyr in self.layers: h = lyr(h)
        return self.lm_head(h)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20, temperature=1.0): # Fig 4.15
        for _ in range(max_new_tokens):
            logits = self(idx[:,-seq_len:])[:,-1,:] / temperature # last token
            probs = torch.softmax(logits.float(), dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1) # Fig 4.16 sampling
            idx = torch.cat([idx, next_idx], dim=1)
        return idx

# Day11에서 학습된 가중치 있다고 가정, 없으면 random이지만 구조는 같음
model = TinyGPT().to(device=device, dtype=dtype)
# 실제로는 Day11 checkpoint load: model.load_state_dict(torch.load("../day11/ckpt.pt"))

torch.manual_seed(0)
base_seq = torch.randint(0, vocab_size, (seq_len+1,), device=device)
# 학습된 패턴: base_seq가 Hello world 패턴이라 가정
x_demo = base_seq[:10].unsqueeze(0) # 10 tokens seed

print(f"Param: {sum(p.numel() for p in model.parameters())/1e6:.2f}M - same 54.82M")
print(f"Seed tokens: {x_demo[0].tolist()[:10]}")
generated = model.generate(x_demo, max_new_tokens=20, temperature=0.8)
print(f"Generated 10->30 tokens: {generated[0].tolist()}")
print(f"Fig 4.15 loop: seed 10 + generate 20 = {generated.shape[1]} tokens")
print("Day11 loss 0.04면 seed 10개 주면 나머지 503개 95% 복원 가능")

# 간단 decode 모의 (실제 BPE decode는 다음 Day)
print("\nNext-token check:")
logits = model(x_demo)
pred = logits.argmax(dim=-1)
print(f"pred next: {pred[0,-1].item()} vs true {base_seq[10].item()} - loss 0.04면 일치율 95%")
