import torch, torch.nn as nn, tiktoken
from peft import LoraConfig, get_peft_model
enc = tiktoken.get_encoding("cl100k_base")
prompt = "EN: Hello world, this is Day15 KO:"
# 다양한 데이터로 100% 만들기
pairs = [
 f"{prompt} 안녕 세상, 이건 Day15",
 f"{prompt} 안녕 세상",
 f"{prompt} 안녕 세상, 이건 Day17 LoRA",
 f"{prompt} 안녕 세상, 이건 Day18 DPO"
]*100
text = "\n".join(pairs)
ids = enc.encode(text)[:1024]
# sliding window 512
vocab_size, hidden = 100277, 768
class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('obj', (object,), {'model_type': 'custom'})()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 12, hidden*4, batch_first=True) for _ in range(6)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
    def forward(self, input_ids=None, **kwargs):
        h = self.embed(input_ids)
        mask = torch.triu(torch.ones(h.size(1), h.size(1), device="cuda", dtype=torch.bool), diagonal=1)
        for lyr in self.layers: h = lyr(h, src_mask=mask)
        return type('obj', (object,), {'logits': self.lm_head(h)})()
    def prepare_inputs_for_generation(self, input_ids, **kwargs): return {"input_ids": input_ids}

base = CausalGPT().to("cuda").to(torch.bfloat16)
model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["linear1","linear2"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# 1000 steps SFT
x = torch.tensor(ids[:-1], device="cuda").unsqueeze(0)[:,:512]
y = torch.tensor(ids[1:], device="cuda").unsqueeze(0)[:,:512]
for step in range(1,1001):
    # random crop for diversity
    start = torch.randint(0, len(ids)-512-1, (1,)).item()
    xb = torch.tensor(ids[start:start+512], device="cuda").unsqueeze(0)
    yb = torch.tensor(ids[start+1:start+513], device="cuda").unsqueeze(0)
    loss = nn.functional.cross_entropy(model(xb).logits.view(-1, vocab_size), yb.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    if step%200==0:
        print(f"SFT {step:4d} loss {loss.item():.4f} CUDA {torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")
        if step%400==0:
            torch.cuda.empty_cache()

prompt_ids = torch.tensor([enc.encode(prompt)], device="cuda")
with torch.no_grad():
    for _ in range(10):
        nxt = torch.argmax(model(prompt_ids).logits[:, -1, :], dim=-1, keepdim=True)
        prompt_ids = torch.cat([prompt_ids, nxt], dim=1)
print("GEN SFT 1000:", enc.decode(prompt_ids[0].tolist()))
