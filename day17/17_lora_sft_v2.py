import torch, torch.nn as nn, tiktoken
from peft import LoraConfig, get_peft_model
enc = tiktoken.get_encoding("cl100k_base")
pairs = [("Hello world, this is Day15", "안녕 세상, 이건 Day15")]*200
text = " ".join([f"EN: {en} KO: {ko}" for en,ko in pairs])
ids = enc.encode(text)[:513]
x,y = torch.tensor(ids[:-1], device="cuda").unsqueeze(0), torch.tensor(ids[1:], device="cuda").unsqueeze(0)

vocab_size, hidden, n_layers = 100277, 768, 6
class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('obj', (object,), {'model_type': 'custom'})()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 12, hidden*4, batch_first=True) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
    def forward(self, input_ids=None, **kwargs):
        h = self.embed(input_ids)
        mask = torch.triu(torch.ones(h.size(1), h.size(1), device="cuda", dtype=torch.bool), diagonal=1)
        for lyr in self.layers: h = lyr(h, src_mask=mask)
        return type('obj', (object,), {'logits': self.lm_head(h)})()
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

base = CausalGPT().to("cuda").to(torch.bfloat16)
model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["linear1","linear2"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
for step in range(1,501):
    loss = nn.functional.cross_entropy(model(x).logits.view(-1, vocab_size), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    if step%100==0:
        print(f"{step} | loss {loss.item():.4f} | {torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

prompt = torch.tensor([enc.encode("EN: Hello world, this is Day15 KO:")], device="cuda")
with torch.no_grad():
    for _ in range(5):
        nxt = torch.argmax(model(prompt[:,-512:]).logits[:,-1:], dim=-1)
        prompt = torch.cat([prompt, nxt], dim=1)
print("GEN 500steps:", enc.decode(prompt[0].tolist()))
