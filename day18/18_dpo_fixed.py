import torch, torch.nn as nn, tiktoken, os
from peft import LoraConfig, get_peft_model
enc = tiktoken.get_encoding("cl100k_base")
prompt = "EN: Hello world, this is Day15 KO:"
# Day17에서 500 steps 학습한 가중치가 없으므로 SFT warmup 200 steps 먼저
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
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

# SFT warmup from scratch + DPO
base = CausalGPT().to("cuda").to(torch.bfloat16)
model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["linear1","linear2"], lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"))
model.print_trainable_parameters()

# 1) SFT 200 steps to get 안녕 세
pairs_sft = [f"{prompt} 안녕 세상, 이건 Day15"]*100
text = " ".join(pairs_sft)
ids = enc.encode(text)[:513]
x,y = torch.tensor(ids[:-1], device="cuda").unsqueeze(0), torch.tensor(ids[1:], device="cuda").unsqueeze(0)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(1,201):
    loss = nn.functional.cross_entropy(model(x).logits.view(-1, vocab_size), y.view(-1))
    loss.backward(); opt.step(); opt.zero_grad()
    if step%100==0:
        print(f"SFT {step} loss {loss.item():.4f} {torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

# 2) DPO 200 steps with reference model frozen
import copy
ref_model = copy.deepcopy(model)
ref_model.eval()
for p in ref_model.parameters(): p.requires_grad = False

def pad_batch(texts, max_len=64):
    ids_list = [enc.encode(t)[:max_len] for t in texts]
    max_l = max(len(x) for x in ids_list)
    return torch.tensor([x + [0]*(max_l-len(x)) for x in ids_list], device="cuda")

def dpo_loss_full(chosen, rejected, beta=0.1):
    # policy logp
    cp_logits = model(chosen).logits
    rp_logits = model(rejected).logits
    # ref logp
    with torch.no_grad():
        ref_cp_logits = ref_model(chosen).logits
        ref_rp_logits = ref_model(rejected).logits
    # per sequence mean CE (ignore pad 0)
    def seq_logp(logits, ids):
        ce = nn.functional.cross_entropy(logits.view(-1, vocab_size), ids.view(-1), reduction='none', ignore_index=0)
        # sum / count non-pad, negative CE = logp
        mask = (ids.view(-1)!=0).float()
        return -(ce*mask).sum() / mask.sum().clamp(min=1)
    # batch mean
    logp_cp = torch.stack([seq_logp(cp_logits[i:i+1], chosen[i:i+1]) for i in range(chosen.size(0))]).mean()
    logp_rp = torch.stack([seq_logp(rp_logits[i:i+1], rejected[i:i+1]) for i in range(rejected.size(0))]).mean()
    logp_ref_cp = torch.stack([seq_logp(ref_cp_logits[i:i+1], chosen[i:i+1]) for i in range(chosen.size(0))]).mean()
    logp_ref_rp = torch.stack([seq_logp(ref_rp_logits[i:i+1], rejected[i:i+1]) for i in range(rejected.size(0))]).mean()
    # DPO with reference
    diff = (logp_cp - logp_ref_cp) - (logp_rp - logp_ref_rp)
    return -torch.log(torch.sigmoid(beta*diff))

opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
chosen_texts = [f"{prompt} 안녕 세상, 이건 Day15", f"{prompt} 안녕 세상"]*2
rejected_texts = [f"{prompt} Hello world, this is Day15", f"{prompt} Hello world"]*2
for step in range(1,201):
    c_ids = pad_batch(chosen_texts)
    r_ids = pad_batch(rejected_texts)
    loss = dpo_loss_full(c_ids, r_ids, beta=0.1)
    loss.backward(); opt.step(); opt.zero_grad()
    if step%50==0:
        print(f"DPO {step} loss {loss.item():.4f} CUDA {torch.cuda.memory_allocated()/1024**2:.0f}MB Peak {torch.cuda.max_memory_allocated()/1024**2:.0f}MB")

prompt_ids = torch.tensor([enc.encode(prompt)], device="cuda")
with torch.no_grad():
    for _ in range(8):
        nxt = torch.argmax(model(prompt_ids).logits[:, -1, :], dim=-1, keepdim=True)
        prompt_ids = torch.cat([prompt_ids, nxt], dim=1)
print("GEN DPO FIXED:", enc.decode(prompt_ids[0].tolist()))
