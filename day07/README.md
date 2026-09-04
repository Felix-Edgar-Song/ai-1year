# Day7 - Scale 43M 5.7s 0.0042 + wandb Phase2 Start

## 🇰🇷 한국어
- Param 0.66M→43.30M (hidden 128→768, layers 4→6, ff 768*4)
- 결과: loss 6.4319→0.0042 200 steps 5.7s CUDA 322.8MB Peak 523.8MB
- 67s vs 5.7s 차이: seq 128로 작음, seq 512면 67s 근처
- bf16 50%로 43M이 523MB만 사용, float32면 1GB+
- wandb 첫 연결 성공: https://wandb.ai/lightel-test/ai-1year/runs/ycuiiilk
- 의미: Phase2 시작, 스케일업 경험

## 🇺🇸 English
- Param 0.66M→43.30M (hidden 128→768, layers 4→6, ff 768*4)
- Result: loss 6.4319→0.0042 200 steps 5.7s CUDA 322.8MB Peak 523.8MB
- 67s vs 5.7s diff: seq 128 small, seq 512 would be ~67s
- bf16 50% enables 43M in 523MB, float32 would be 1GB+
- wandb first sync success
- Insight: Phase2 start, scale-up experience

## Log
```
Param: 43.30M
1 | 6.4319 | 3.1s | 322.8MB | 495.2MB
20 | 0.0152 | 3.3s
200 | 0.0042 | 5.7s | 322.8MB | 523.8MB
GPU: RTX 5060 Ti
wandb: loss 0.00421
```
