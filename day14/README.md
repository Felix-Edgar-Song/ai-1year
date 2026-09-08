# Day14 - Causal Mask 1 Line 15% -> 100% Match Real==Gen

## 🇰🇷 한국어

### 결과 요약
- **모델**: Param 54.82M Fig 4.3 same, CausalGPT
- **학습**: 100 steps 9.1326 -> 0.0016 (99.8%) 494MB Peak 673MB 25 steps 0.0336
- **생성**: Short 10->30 100%, Long 492->512 100% Real [961, 4579, 1571, 5709, 2185] == Gen 완전 일치
- **환경**: Ubuntu native (WSL2 vLLM 불가로 이전), RTX 5060 Ti 16GB, Driver 595.84 CUDA 13.2, Torch 2.11.0+cu128, ai-env

### 책 연결 - 왜 15% -> 100%가 됐나

**Chapter 3 Section 3.3 Figure 3.7 Bidirectional Self-Attention (어제 실패)**
- 정의: 모든 토큰 서로 봄 mask 없음, BERT 구조, 512개 전체로 예측 = 미래 치팅
- 내 Day13 `lyr(h)`가 이 그림 - loss 0.047 치팅, 생성 15% [961,4579,1571]까지만

**Chapter 4 Section 4.4 Figure 4.5 Causal Mask (오늘 성공)**
- 정의: 미래 가림 `triu(diagonal=1) = -inf`, 과거만 봄, GPT/Llama 필수
- 내 Day14 `lyr(h, src_mask=causal_mask)` 1줄이 이 그림 - loss 0.0016 진짜 학습, 생성 100%

**Chapter 4 Section 4.6 Figure 4.15 Autoregressive Generation Loop (오늘 증적)**
- 정의: `idx[:, -seq_len:]` 최근 context로 next token 반복 생성
- 내 로그 `Real==Gen [961][4579][1571][5709][2185]`가 이 그림 증적

## 🇺🇸 English

### Summary
- **Model**: Param 54.82M, Training 9.1326 -> 0.0016 99.8% 494MB Peak 673MB
- **Generation**: Short 100%, Long 100% Real==Gen [961,4579,1571,5709,2185]
- **Env**: Ubuntu native, RTX 5060 Ti 16GB, 595.84 CUDA 13.2, Torch 2.11.0+cu128, ai-env

### Book Link
- Ch 3 Sec 3.3 Fig 3.7 Bidirectional - no mask, Day13 15% failure
- Ch 4 Sec 4.4 Fig 4.5 Causal Mask - triu, Day14 100% success
- Ch 4 Sec 4.6 Fig 4.15 Generation Loop - 100% proof

## Log

```
Param: 54.82M Fig 4.5 causal
1 | 9.1326 | 494MB | 603MB
25 | 0.0336 | 673MB
100 | 0.0016 | 494MB | 673MB
Short 10->30 100%
Long 492->512 100% Real==Gen
CUDA 105MiB idle[961][4579][1571][5709][2185]
```
## Files

- `14_causal_gpt.py` - English comments only, Fig 4.5 causal mask implementation
- `log.txt` - training log 9.13->0.0016 100% Match 494MB/673MB

## References

- Raschka, S. (2024). Build a Large Language Model (From Scratch). Manning.
    - Chapter 3 Section 3.3 Figure 3.7 - Bidirectional Self-Attention
    - Chapter 4 Section 4.4 Figure 4.5 - Causal Mask
    - Chapter 4 Section 4.6 Figure 4.15 - Autoregressive Generation

