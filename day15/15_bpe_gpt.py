import torch, torch.nn as nn, tiktoken
# Fig 5.8 BPE tokenizer - real English tokens
enc = tiktoken.get_encoding("cl100k_base") # GPT-4 tokenizer 100k vocab
text = "Hello world, this is Day15 causal GPT with real BPE tokenizer."
ids = enc.encode(text) # str -> List[int]
print(f"Text: {text}")
print(f"BPE ids: {ids} len={len(ids)}")
print(f"Decode check: {enc.decode(ids)}")

# Model same as Day14 Fig 4.5 causal
vocab_size = 100277 # cl100k_base vocab size
hidden, n_layers = 768, 6
device = "cuda"

class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden) # Fig 2.10
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 12, hidden*4, batch_first=True) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False) # Fig 4.3
    def forward(self, x):
        h = self.embed(x)
        # FIX: dynamic causal mask based on actual seq_len = x.size(1) - Fig 4.5
        seq_len = x.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1) # Fig 4.5 dynamic
        for lyr in self.layers:
            h = lyr(h, src_mask=causal_mask)
        return self.lm_head(h)
    @torch.no_grad()
    def generate(self, idx, max_new_tokens=20): # Fig 4.15
        for _ in range(max_new_tokens):
            seq_len = idx.size(1)
            # use last 512 if longer, but mask size matches actual input
            logits = self(idx[:, -512:])[:, -1, :]
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

model = CausalGPT().to(device).to(torch.bfloat16)
# FIX: make 512 exactly by repeating enough
repeat = 512 // len(ids) + 2
base_seq = torch.tensor(ids * repeat, device=device)[:513] # 513 ensure 512 input
x, y = base_seq[:-1].unsqueeze(0), base_seq[1:].unsqueeze(0) # [1,512] now
print(f"x shape {x.shape} y shape {y.shape} - should be [1,512]")

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(1, 101):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 25 == 0:
        mem = torch.cuda.memory_allocated()/1024**2
        peak = torch.cuda.max_memory_allocated()/1024**2
        print(f"{step:3d} | loss {loss.item():.4f} | CUDA {mem:.0f}MB | Peak {peak:.0f}MB")

# Generate test
prompt_ids = enc.encode("Hello world, this is")
prompt = torch.tensor([prompt_ids], device=device)
gen_ids = model.generate(prompt, max_new_tokens=10)[0].tolist()
print(f"Prompt: {enc.decode(prompt_ids)}")
print(f"Gen: {enc.decode(gen_ids)}")
print(f"Gen ids tail: {gen_ids[len(prompt_ids):]}")
# Expected continuation: " Day15 causal GPT..."
print(f"Expected: {enc.decode(ids[4:])}") # after "Hello world, this is"
