import torch
import torch.nn as nn

# Model: 375M CausalGPTCustom 757MB bf16 / 모델: 375M 757MB bf16
# torch.compile: reduce-overhead mode / 오버헤드 감소 모드
# bfloat16: halves memory / 메모리 절반

class SimpleGPT(nn.Module):
    def __init__(self, vocab=50257, d_model=1024, n_layer=24):
        super().__init__()
        self.wte = nn.Embedding(vocab, d_model) # token embedding / 토큰 임베딩
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model, 8, batch_first=True) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model) # final norm / 최종 정규화
        self.lm_head = nn.Linear(d_model, vocab, bias=False) # language modeling head / 언어 모델링 헤드
    
    def forward(self, x):
        # x: [B, T] -> [B, T, d_model] / 입력 [배치, 시간] -> [배치, 시간, 차원]
        h = self.wte(x)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        return self.lm_head(h) # logits / 로짓

# Load / 로드
model = SimpleGPT()
model = model.to(dtype=torch.bfloat16, device='cuda') # bf16 conversion / bf16 변환
model = torch.compile(model, mode="reduce-overhead") # compile for 400MiB / 400MiB 위해 컴파일
print(f"Compiled model - VRAM: {torch.cuda.memory_allocated()/1024**2:.1f}MiB") # VRAM check / VRAM 확인

# Generate test / 생성 테스트
with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16): # no grad + autocast / 그래디언트 없이 + 자동 캐스팅
    x = torch.randint(0, 50257, (1, 32), device='cuda') # dummy input / 더미 입력
    out = model(x)
    print(f"Output shape: {out.shape} / 출력 모양") # [B, T, vocab] / [배치, 시간, 어휘]
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated()/1024**2:.1f}MiB / 최대 VRAM") # peak check / 최대 확인

# nvidia-smi comparison / 비교
# Expected: 902MiB (Day24) -> 608MiB (Day25 vLLM) -> 400MiB (Day26 torch.compile) / 예상
