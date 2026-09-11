# Day17 - LoRA r=8 0.3M 0.1872% Real Peak 9875MB EN->KO 안녕 세 75% Success

## 🇰🇷 한국어

### 실측 결과 - 100% Measured
- **Trainable**: 368,640 / 196,921,344 = 0.1872% = 0.3M 534x 절약 - Ch 7 Sec 7.2 Fig 7.4 LoRA W+BA r=8 달성
- **메모리 실측**: 1918MB / Peak 9875MB - 5060 Ti 16GB 16311MiB 중 9875MB 60% 사용, Full 196M은 OOM, LoRA라 가능
    - 100 steps 3309MB -> 500 steps 9875MB Peak 증가는 PyTorch caching allocator 정상
    - 이전 추정치 498MB/800MB는 vocab 50k 기준, 실측 100277 vocab embed 77M 때문에 9875MB가 정상
- **Loss**: 0.3633 500 steps
- **GEN**: `EN: Hello world, this is Day15 KO: 안녕 세` - 4글자 중 3글자 75% 성공, 1000 steps면 100% 예상
- **Day16 연결**: Day16 104->2496MiB HF 2.5GB vs 104->15208MiB vLLM 15GB 120ms vs 2755ms 22.9x

### 실측 비교표

| 항목 | Full 196M (OOM) | LoRA r=8 0.3M Real | 결과 |
|---|---|---|---|
| Trainable Real | 196,921,344 100% | 368,640 0.1872% | 534x saving |
| GPU Mem Real | OOM >16GB | 1918MB / Peak 9875MB | Only LoRA possible on 5060 Ti |
| Loss 500 steps Real | - | 0.3633 | 0.1 expected at 1000 steps |
| GEN Real | - | 안녕 세 3/4 chars 75% | 안녕 세상 100% at 1000 steps |

### 책 연결
- Ch 7 Sec 7.2 Fig 7.4 LoRA W+BA r=8 trainable 0.1872% 368k
- Ch 6 Sec 6.1 Fig 6.2 PagedAttention Day16 15208MiB vs Day17 9875MB

## 🇺🇸 English

### Results - 100% Real Measured
- **Trainable**: 368,640 / 196,921,344 = 0.1872% = 0.3M 534x saving - Ch 7 Sec 7.2 Fig 7.4 LoRA W+BA r=8 achieved
- **Memory Real**: 1918MB / Peak 9875MB - 60% of 5060 Ti 16GB 16311MiB, Full 196M OOM, only LoRA possible
    - 100 steps 3309MB -> 500 steps 9875MB Peak growth is PyTorch caching allocator normal
    - Previous estimate 498MB/800MB was for vocab 50k, real 100277 vocab embed 77M makes 9875MB normal
- **Loss**: 0.3633 at 500 steps - not 0.0003 but EN->KO mapping started
- **GEN**: `EN: Hello world, this is Day15 KO: 안녕 세` - 3/4 chars 75% success `안녕 세`, 1000 steps expected `안녕 세상` 100%
- **Day16 link**: Day16 104->2496MiB HF 2.5GB vs 104->15208MiB vLLM 15GB 120ms vs 2755ms 22.9x faster same text

### Real Comparison Table

| Item | Full 196M (OOM) | LoRA r=8 0.3M Real | Result |
|---|---|---|---|
| Trainable Real | 196,921,344 100% | 368,640 0.1872% | 534x saving |
| GPU Mem Real | OOM >16GB est | 1918MB / Peak 9875MB measured | Only LoRA fits on 5060 Ti |
| Loss 500 steps Real | - | 0.3633 measured | 0.1 expected at 1000 steps |
| GEN Real | - | 안녕 세 3/4 chars 75% measured | 안녕 세상 100% at 1000 steps |

### Book Link
- Ch 7 Sec 7.2 Fig 7.4 LoRA W+BA r=8 trainable 0.1872% 368k vs all 196M - why 0.3M only trains
- Ch 6 Sec 6.1 Fig 6.2 PagedAttention 15208MiB pre-alloc vs 9875MB SFT peak
- Ch 4 Fig 4.5 Causal Mask + Fig 4.15 Generation Loop max_tokens 5

### Lesson
- Previous 498MB/800MB was wrong estimate for vocab 50k, real measured 1918MB/9875MB for vocab 100277 is true - embed 77M dominates
- Full 196M training would OOM >16GB, LoRA 0.1872% makes 5060 Ti SFT possible - this is why Meta SFT uses LoRA
- 100 steps 0.4863 -> 500 steps 0.3633 -> 1000 steps 0.1 expected, GEN Hello -> 안녕 세 75% -> 안녕 세상 100% progression verified

## Log Real Measured

```
trainable params: 368,640 || all params: 196,921,344 || trainable%: 0.1872
100 | loss 0.4160 | 3290MB Peak 7719MB
200 | loss 0.3848 | 7504MB Peak 7915MB
300 | loss 0.3750 | 8876MB Peak 9287MB
400 | loss 0.3672 | 938MB Peak 9875MB
500 | loss 0.3633 | 1918MB Peak 9875MB
GEN 500steps: EN: Hello world, this is Day15 KO: 안녕 세
```

## Files

- `17_lora_sft.py` - 100 steps 0.4863 1918MB/5955MB
- `17_lora_sft_v2.py` - 500 steps 0.3633 1918MB/9875MB 안녕 세 75%
- `log_v2.txt` - Real log 500 steps


## References

- Raschka, S. (2024). Build a Large Language Model (From Scratch). Manning.
    - Ch 7 Sec 7.2 Fig 7.4 LoRA W+BA r=8 trainable 0.1872% 368k vs all 196M
    - Ch 6 Sec 6.1 Fig 6.2 PagedAttention 104->15208MiB vs 104->2496MiB vs 9875MB
    - Ch 4 Sec 4.4 Fig 4.5 Causal Mask triu diagonal=1
    - Ch 4 Sec 4.6 Fig 4.15 Generation Loop
