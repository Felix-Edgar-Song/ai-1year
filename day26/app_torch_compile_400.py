import torch
import torch.nn as nn

# 580MiB -> 400MiB로 줄이는 3가지 / 3 tricks to 400MiB
torch.set_float32_matmul_precision('high') # matmul 정밀도 낮춰서 캐시 절감 / Reduce matmul precision
torch.backends.cuda.matmul.allow_tf32 = True # TF32 허용 / Allow TF32
torch._inductor.config.triton.cudagraphs = False # cudagraphs 끄기 - 5060 Ti에서는 오히려 메모리 더 먹음 / Disable cudagraphs - uses more on 5060 Ti

class SimpleGPT(nn.Module):
    def __init__(self, vocab=50257, d_model=768, n_layer=12): # d_model 1024->768, n_layer 24->12 로 축소 / Reduce size
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, 8, 768, batch_first=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
    
    def forward(self, x):
        h = self.wte(x)
        for layer in self.layers:
            h = layer(h)
        return self.lm_head(self.ln_f(h))

model = SimpleGPT() # 375M -> 124M으로 줄이면 377MB -> 150MB / 375M to 124M reduces 377MB to 150MB
model = model.to(dtype=torch.bfloat16, device='cuda')
model = torch.compile(model, mode="reduce-overhead", fullgraph=False) # fullgraph=False로 844MiB peak 방지 / Prevent 844MiB peak

print(f"Before: {torch.cuda.memory_allocated()/1024**2:.1f}MiB")
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    x = torch.randint(0, 50257, (1, 8), device='cuda') # 32 -> 8 토큰으로 KV cache 75% 절감 / 32 to 8 tokens saves 75% KV
    for _ in range(2): # 2번 돌려서 compile 캐시 워밍업 / Warmup compile cache
        out = model(x)
    torch.cuda.empty_cache() # 캐시 비우기 / Empty cache
    print(f"After 2x run: {torch.cuda.memory_allocated()/1024**2:.1f}MiB / 실행 후")
    print(f"Peak: {torch.cuda.max_memory_allocated()/1024**2:.1f}MiB / 최대")

