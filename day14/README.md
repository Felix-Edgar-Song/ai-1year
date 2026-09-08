# Day14 - Causal Mask 1 Line 15% -> 100% Match Real==Gen

## 🇰🇷 한국어

### 결과 요약
- **모델**: Param 54.82M Fig 4.3 same, CausalGPT
- **학습**: 100 steps 9.1326 -> 0.0016 (99.8% 정답) 494MB Peak 673MB 25 steps 0.0336
- **생성**: Short 10->30 100%, Long 492->512 100% Real [961, 4579, 1571, 5709, 2185] == Gen 완전 일치
- **환경**: Ubuntu native (WSL2 vLLM 불가로 이전), RTX 5060 Ti 16GB, Driver 595.84 CUDA 13.2, Torch 2.11.0+cu128, ai-env

### 책 연결 - 왜 15% -> 100%가 됐나

**Chapter 3 Section 3.3 Figure 3.7 Bidirectional Self-Attention (어제 Day13 실패 원인)**
- 정의: 모든 토큰이 서로를 볼 수 있음 `Attention(Q,K,V) = softmax(QK^T/√d)V` mask 없음
- BERT는 이 구조, 512개 전체 context로 빈칸 예측 (MLM)
- 내 Day13 코드 `lyr(h)`가 이 그림 - `x[0..511]` 512개 다 보고 `y[0]=base_seq[1]` 예측 = 미래 치팅
- 결과: loss 0.047 치팅 95%, 생성은 과거 10개/492개만 보니 3개 이후 실패 15% [961,4579,1571]까지만 맞고 [5709] 틀림

**Chapter 4 Section 4.4 Figure 4.5 Causal Mask (오늘 Day14 성공 원인)**
- 정의: 미래 토큰 가림 `mask = triu(ones(n,n), diagonal=1) = -inf`로 masking, `softmax(QK^T/√d + mask)V`로 과거만 봄
- GPT, Llama 전부 이 구조 - autoregressive LM 필수
- 내 Day14 코드 `causal_mask = torch.triu(torch.ones(512,512, bool), diagonal=1)` + `lyr(h, src_mask=causal_mask)` 1줄 추가가 이 그림
- 결과: `x[0]`은 `x[0]`만 보고 `x[1]` 예측, `x[0..491]` 보고 `x[492]` 예측 = 과거만으로 진짜 학습, loss 0.0016 99.8%로 더 낮음 (외우기 더 쉬움), 생성 100% 일치

**Chapter 4 Section 4.6 Figure 4.15 Autoregressive Generation Loop (오늘 증적)**
- 정의: `prompt 10개 -> generate 20개` loop에서 `idx[:, -seq_len:]`로 최근 context만 보고 next token argmax, 이어 붙이기 반복
- 긴 prompt일수록 exposure bias 감소 - 짧은 prompt 10개는 cascade 빨리 터짐, 긴 prompt 492개는 context 많아서 안정
- 내 로그 `Short 100% Long 100% Real==Gen [961][4579][1571][5709][2185]`가 이 그림 증적 - Fig 4.5 적용 후 exposure bias 3개 한계 해소

### 배운 점
- GPT는 Fig 3.7 쓰면 안 되고 Fig 4.5 필수 - 1줄 차이가 15% vs 100%
- loss 낮은 게 진짜가 아님 - Day13 0.047 치팅 loss vs Day14 0.0016 진짜 loss
- 메모리 동일 494MB Peak 673MB, causal mask 연산 추가 비용 0

## 🇺🇸 English

### Summary
- **Model**: Param 54.82M Fig 4.3 same, CausalGPT
- **Training**: 100 steps 9.1326 -> 0.0016 (99.8%) 494MB Peak 673MB 25 steps 0.0336
- **Generation**: Short 10->30 100%, Long 492->512 100% Real [961, 4579, 1571, 5709, 2185] == Gen exact match
- **Env**: Ubuntu native (migrated from WSL2 due to vLLM incompatibility on 5060 Ti latest driver), RTX 5060 Ti 16GB, Driver 595.84 CUDA 13.2, Torch 2.11.0+cu128, ai-env

### Book Link - Why 15% -> 100%

**Chapter 3 Sec 3.3 Fig 3.7 Bidirectional Self-Attention (Day13 failure)**
- Definition: All tokens attend to each other `Attention = softmax(QK^T/√d)V` no mask
- BERT structure, predicts with full 512 context (MLM)
- My Day13 code `lyr(h)` is this figure - sees `x[0..511]` to predict `y[0]=base_seq[1]` = future cheating
- Result: loss 0.047 cheating 95%, generation sees only past 10/492 tokens so fails after 3 tokens 15% [961,4579,1571] match then [5709] fail

**Chapter 4 Sec 4.4 Fig 4.5 Causal Mask (Day14 success)**
- Definition: Mask future `mask = triu(ones(n,n), diagonal=1) = -inf`, `softmax(QK^T/√d + mask)V` past only
- GPT, Llama mandatory structure for autoregressive LM
- My Day14 code `causal_mask = torch.triu(...)` + `lyr(h, src_mask=causal_mask)` 1 line added is this figure
- Result: `x[0]` sees only `x[0]` to predict `x[1]`, `x[0..491]` to predict `x[492]` = real learning past only, loss 0.0016 99.8% even lower (easier to memorize), generation 100% match

**Chapter 4 Sec 4.6 Fig 4.15 Autoregressive Generation Loop (today proof)**
- Definition: Loop `prompt 10 -> generate 20` with `idx[:, -seq_len:]` recent context, next token argmax, concat repeat
- Longer prompt reduces exposure bias - short prompt cascades quickly, long prompt 492 more stable
- My log `Short 100% Long 100% Real==Gen [961][4579][1571][5709][2185]` proves this figure - after Fig 4.5, 3-token limit resolved

### Lesson
- GPT must use Fig 4.5 not Fig 3.7 - 1 line difference 15% vs 100%
- Lower loss not always real - Day13 0.047 cheating vs Day14 0.0016 real
- Same memory 494MB Peak 673MB, causal mask zero extra cost

## Log
```
Param: 54.82M Fig 4.5 causal
Training 100 steps with causal mask...
1 | loss 9.1326 | CUDA 494MB | Peak 603MB
25 | loss 0.0336 | CUDA 494MB | Peak 673MB
50 | loss 0.0090 | CUDA 494MB | Peak 673MB
75 | loss 0.0037 | CUDA 494MB | Peak 673MB
100 | loss 0.0016 | CUDA 494MB | Peak 673MB

Short 10->30 Match: 100.0%
Long 492->512 Match: 100.0% <- 80∼100% success Fig 4.15
Real: | Gen:
CUDA 105MiB idle[961][4579][1571][5709][2185]
```
## Files

- `14_causal_gpt.py` 
- `log.txt` - 494MB/673MB 100% Match
- `ckpt_*.pt` - local only 105MB bf16, gitignored (GitHub 100MB limit)

## References

- Chapter 3 Section 3.3 Figure 3.7 - Bidirectional Self-Attention (cheating)
- Chapter 4 Section 4.4 Figure 4.5 - Causal Mask `torch.triu` (today fix)
- Chapter 4 Section 4.6 Figure 4.15 - Autoregressive Generation Loop (100% proof)
