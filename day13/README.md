# Day13 - Ckpt 105MB Short 30% Long 15% causal mask 없음 증적

## 🇰🇷 한국어
- Param 54.82M 100 steps 9.14->0.0471 495MB Peak 678MB ckpt 105MB bf16
- 결과: Short 10->20 30% [4190,5717,6798] 3개 일치, Long 492->512 15% [961,4579,1571] 3개 일치
- 원인: TransformerEncoderLayer는 causal mask 없음 Fig 3.7 bidirectional 치팅, Fig 4.5 causal mask 필요
- loss 0.047은 미래 보면서 치팅한 95%, 생성은 과거만 보니 3개 이후 cascade 실패 exposure bias
- 의미: GPT는 causal mask 필수, 105MB bf16 저장 219MB float32 절반 절약
- wandb 없이도 pred 4190 vs true 4190으로 95% 증적

## 🇺🇸 English
- Param 54.82M 100 steps 9.14->0.0471 495MB ckpt 105MB bf16
- Short 30% Long 15% both 3 tokens match then fail - no causal mask Fig 3.7 vs Fig 4.5
- Loss 0.047 cheating with bidirectional, generation fails after 3 due to exposure bias

## Log
```
25 | 0.0500 | 495MB
100 | 0.0471 | 495MB
Saved 105M
Short 30%
Long 15% 3 match then fail
CUDA 131MiB idle[4190][5717][6798][961][4579][1571]
```
