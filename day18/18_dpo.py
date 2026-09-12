import torch, torch.nn as nn, tiktoken
from peft import LoraConfig, get_peft_model
enc = tiktoken.get_encoding("cl100k_base")
prompt = "EN: Hello world, this is Day15 KO:"
# chosen vs rejected - same prompt, different answer
pairs = [(f"{prompt} 안녕 세상, 이건 Day15", f"{prompt} Hello world, this is Day15")]*50
vocab_size, hidden = 100277, 768

class CausalGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type('obj', (object,), {'model_type': 'custom', 'hidden_size': hidden})()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden, 12, hidden*4, batch_first=True) for _ in range(6)])
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
model.print_trainable_parameters() # 368k 0.1872%

def pad_batch(ids_list, pad_id=0, max_len=64):
    # ids_list: list of list[int] different lengths
    max_l = max(len(x) for x in ids_list)
    max_l = min(max_l, max_len)
    padded = []
    for x in ids_list:
        x = x[:max_l]
        padded.append(x + [pad_id]*(max_l - len(x)))
    return torch.tensor(padded, device="cuda")

def dpo_loss(chosen_ids, rejected_ids, beta=0.1):
    # ignore pad in loss
    cl = model(chosen_ids).logits
    rl = model(rejected_ids).logits
    # cross entropy per token, mask pad 0
    closs_t = nn.functional.cross_entropy(cl.view(-1, vocab_size), chosen_ids.view(-1), reduction='none', ignore_index=0)
    rloss_t = nn.functional.cross_entropy(rl.view(-1, vocab_size), rejected_ids.view(-1), reduction='none', ignore_index=0)
    closs = closs_t.view(chosen_ids.shape).mean(1)
    rloss = rloss_t.view(rejected_ids.shape).mean(1)
    # DPO: chosen loss low, rejected loss high -> rloss - closs large
    loss = -torch.log(torch.sigmoid(beta * (rloss - closs))).mean()
    return loss

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
for step in range(1,201):
    # batch 2
    batch_pairs = pairs[(step*2)%len(pairs):(step*2)%len(pairs)+2]
    c_texts = [p[0] for p in batch_pairs]
    r_texts = [p[1] for p in batch_pairs]
    c_ids = pad_batch([enc.encode(t) for t in c_texts])
    r_ids = pad_batch([enc.encode(t) for t in r_texts])
    loss = dpo_loss(c_ids, r_ids)
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 50 == 0:
        mem = torch.cuda.memory_allocated()/1024**2
        peak = torch.cuda.max_memory_allocated()/1024**2
        print(f"{step:3d} | DPO loss {loss.item():.4f} | CUDA {mem:.0f}MB | Peak {peak:.0f}MB | chosen > rejected")

# GEN test
prompt_ids = torch.tensor([enc.encode(prompt)], device="cuda")
with torch.no_grad():
    for _ in range(8):
        nxt = torch.argmax(model(prompt_ids).logits[:, -1, :], dim=-1, keepdim=True)
        prompt_ids = torch.cat([prompt_ids, nxt], dim=1)
print("GEN DPO:", enc.decode(prompt_ids[0].tolist()))
