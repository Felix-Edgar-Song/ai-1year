import torch, torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from configuration_causal_gpt_custom import CausalGPTCustomConfig

class CausalGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(config.n_embd, config.n_head, config.n_embd*4, batch_first=True, norm_first=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    def forward(self, x):
        B,T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        for l in self.layers: h = l(h)
        return self.lm_head(self.ln_f(h))

class CausalGPTCustomForCausalLM(PreTrainedModel):
    config_class = CausalGPTCustomConfig
    def __init__(self, config):
        super().__init__(config)
        self.model = CausalGPT(config)
    def forward(self, input_ids, **kwargs):
        logits = self.model(input_ids)
        return CausalLMOutputWithCrossAttentions(logits=logits)
