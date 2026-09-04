# Day9 - Vocab 32K 91M 10.52->9.73 1150MB

## 🇰🇷 한국어
- Param 43.30M→91.68M (vocab 500→32000 64배, embed 0.38M→24.5M x2)
- 결과: loss 10.5209→9.7316 200 steps 8.0s CUDA 597MB Peak 1150MB
- 시작 loss ln(32000)=10.37 이론과 일치, 안 떨어지는 이유: vocab 64배 어려움 + batch 2 작음 + lr 3e-4 작음
- CUDA +270MB 증가, bf16 50%로 1150MB, float32면 1.7GB OOM
- 시간 7.3s→8.0s 10%만 증가: vocab은 메모리 병목, layers가 시간 병목
- 책 밑바닥부터 배우는 LLM 3장 토크나이저 vocab이 메모리 50% 차지 증적
- wandb: https://wandb.ai/lightel-test/ai-1year/runs/ap8vqd3m loss 불안정 그래프

## 🇺🇸 English
- Param 43.30M->91.68M (vocab 500->32000 64x, embed 0.38M->24.5M x2)
- Result: loss 10.5209->9.7316 200 steps 8.0s CUDA 597MB Peak 1150MB
- Start loss ln(32000)=10.37 matches theory, slow drop: vocab 64x harder + batch 2 small + lr 3e-4 small
- CUDA +270MB, bf16 50% enables 1150MB, float32 would be 1.7GB OOM
- Time 7.3s->8.0s +10%: vocab is memory bottleneck, layers is time bottleneck
- Book: LLM from scratch ch3 vocab dominates 50% memory
- wandb: unstable loss graph shows large vocab training difficulty

## Log
```
Param: 91.68M
1 | 10.5209 | 3.1s | 597MB | 952MB
20 | 10.5004 | 3.5s | 597MB | 1150MB
200 | 9.7316 | 8.0s | 597MB | 1150MB
GPU: RTX 5060 Ti 16GB
Total: 8.1s
```
