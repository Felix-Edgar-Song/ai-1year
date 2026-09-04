# Day8 - Seq 512 7.3s O(n^2) optimization discovered

## 🇰🇷 한국어
- Param 43.30M 유지, seq 128→512 (4배)
- 결과: loss 6.3960→0.0034 200 steps 7.3s CUDA 326MB Peak 704MB fwd 1503ms→4ms
- 67s vs 7.3s 차이: Torch 2.11+cu128 scaled_dot_product_attention이 flash-attention처럼 자동 최적화, 첫 step 컴파일 후 4ms 캐시
- 의미: 최신 GPU/프레임워크는 O(n²)도 375배 빠르게, 16GB에서 43M 512 seq 가능
- wandb: https://wandb.ai/lightel-test/ai-1year/runs/hc4u1qi1 (Day7), https://wandb.ai/lightel-test/ai-1year/runs/hc4u1qi1 (Day8)

## 🇺🇸 English
- Param 43.30M same, seq 128→512 (4x)
- Result: loss 6.3960→0.0034 200 steps 7.3s CUDA 326MB Peak 704MB fwd 1503ms→4ms
- 67s vs 7.3s diff: Torch 2.11+cu128 uses flash-attention internally, 1503ms compile then 4ms cache
- Insight: modern stack optimizes O(n²) 375x, 43M 512 seq fits in 16GB
- wandb: fwd_ms graph shows compile cache hit

## Log

```
Param: 43.30M
1 | 6.3960 | 3.2s | 326MB | 529MB | fwd 1503ms
20 | 0.0106 | 3.5s | 326MB | 705MB | fwd 4ms
200 | 0.0034 | 7.3s | 326MB | 705MB | fwd 4ms
GPU: RTX 5060 Ti
Peak: 704.9MB
```
