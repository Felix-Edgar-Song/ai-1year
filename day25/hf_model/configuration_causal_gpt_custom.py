from transformers import PretrainedConfig
class CausalGPTCustomConfig(PretrainedConfig):
    model_type = "causal_gpt_custom"
    def __init__(self, vocab_size=100277, n_embd=768, n_layer=6, n_head=12, n_positions=512, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_positions = n_positions
