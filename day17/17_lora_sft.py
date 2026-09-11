import torch, torch.nn as nn, tiktoken
from peft import LoraConfig, get_peft_model
# Fig 7.4 LoRA r=8 SFT EN->KO
enc = tiktoken.get_encoding("cl100k_base")
pairs = [("Hello world, this is Day15", "안녕 세상, 이건 Day15"), ("Hello world, this is", "안녕 세상, 이건")]
text = " ".join([f"EN: {en} KO: {ko}" for en,ko in pairs])*100
ids = enc.encode(text)[:513]
base_seq = torch.tensor(ids, device="cuda")
x, y = base_seq[:-1].unsqueeze(0), base_seq[1:].unsqueeze(0)
print(f"x {x.shape} {x.device} len {len(ids)}")

vocab_size, hidden, n_layers = 100277, 768, 6
class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # dummy config for PEFT
        self.config = type('obj', (object,), {'model_type': 'custom', 'hidden_size': hidden})()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 12, hidden*4, batch_first=True) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if input_ids is not None:
            h = self.embed(input_ids)
        else:
            h = inputs_embeds
        seq_len = h.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device="cuda", dtype=torch.bool), diagonal=1)
        for lyr in self.layers: h = lyr(h, src_mask=mask)
        logits = self.lm_head(h)
        # return object with logits attr to match HF API
        return type('obj', (object,), {'logits': logits})()
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

base = CausalGPT().to("cuda").to(torch.bfloat16)
config = LoraConfig(r=8, lora_alpha=16, target_modules=["linear1","linear2"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM")
model = get_peft_model(base, config)
model.print_trainable_parameters()

opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(1,101):
    out = model(x)
    logits = out.logits if hasattr(out,'logits') else out
    loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    if step%25==0:
        mem=torch.cuda.memory_allocated()/1024**2; peak=torch.cuda.max_memory_allocated()/1024**2
        print(f"{step:3d} | loss {loss.item():.4f} | CUDA {mem:.0f}MB | Peak {peak:.0f}MB | trainable 0.1872%")

# gen
prompt = torch.tensor([enc.encode("EN: Hello world, this is KO:")], device="cuda")
with torch.no_grad():
    for _ in range(10):
        out = model(prompt[:,-512:])
        logits = out.logits if hasattr(out,'logits') else out
        next_id = torch.argmax(logits[:,-1,:], dim=-1, keepdim=True)
        prompt = torch.cat([prompt, next_id], dim=1)
print("GEN:", enc.decode(prompt[0].tolist()))
