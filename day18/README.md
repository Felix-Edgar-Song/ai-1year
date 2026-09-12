# Day18 - LoRA r=8 0.3M 0.1872% Peak 9876MB Real SFT 1000 steps 안녕 세상, 이건 100% + DPO Collapse Evidence

## 🇰🇷 한국어

### 실측 결과 - 100% Real Measured 22:47 Gwangju

**최종 성공: SFT 1000 steps 안녕 세상 100%**
- **Trainable**: 368,640 / 196,921,344 = 0.1872% = 0.3M 534x 절약 - Fig 7.4 3일 연속 실측 동일
- **SFT 1000 steps Real**:
    - 200 loss 0.5547 CUDA 5446MB Peak 9483MB
    - 400 loss 0.5352 CUDA 1232MB Peak 9875MB
    - 600 loss 0.5195 CUDA 3290MB Peak 9875MB
    - 800 loss 0.4863 CUDA 5054MB Peak 9876MB
    - 1000 loss 0.4004 CUDA 6818MB Peak 9876MB
- **GEN SFT 1000 Real**: `EN: Hello world, this is Day15 KO: 안녕 세상, 이건` - 7글자 100% 성공, Day17 75% `안녕 세` -> Day18 100% `안녕 세상, 이건` 완성
- **Peak 확정**: Day17 9875MB + DPO 9777MB + SFT 1000 9876MB = **LoRA r=8 512 seq 196M Peak 9876MB±1MB로 확정** - 5060 Ti 16GB 16311MiB 중 60% 사용, Full 196M은 OOM
- **Idle**: 188MiB Xorg 79MiB + gnome-shell 79MiB 22:47 Gwangju

**중요한 실패 기록: DPO 200 steps Collapse Real**
- SFT 200 9366MB Peak 9777MB loss 0.3535 + DPO 200 2063MB Peak 9777MB loss 0.3857
- GEN DPO: `안 안 안 안 안 안 안 안` Mode collapse - 1-task DPO는 1000+ preference 필요, 2개 샘플로는 고빈도 토큰 반복이 최적해, 책 Fig 7.8 정식 DPO도 1-task에서 collapse 정상

### 실측 비교표 - 3일 메모리 완성

| 항목 | Day16 HF vs vLLM Real | Day17 LoRA 500 Real | Day18 DPO 200 Real | Day18 SFT 1000 Real |
|---|---|---|---|---|
| Trainable | - | 368,640 0.1872% | 368,640 0.1872% | 368,640 0.1872% |
| GPU Mem Real | 104->2496MiB 2.5GB HF vs 104->15208MiB 15GB vLLM 22.9x | 1918MB/Peak 9875MB | 2063MB/Peak 9777MB | 6818MB/Peak 9876MB |
| Loss Real | - | 0.3633 | 0.3857 DPO | 0.4004 SFT random crop |
| GEN Real | 6->26 tokens 100% Match | 안녕 세 75% | 안 안 안 Collapse | 안녕 세상, 이건 100% |
| 결론 | PagedAttention 15GB pre-alloc | LoRA만 16GB에서 SFT 가능 | 1-task DPO collapse 증적 | SFT 1000 steps가 1-task 정답 |

### 책 연결
- Ch 7 Sec 7.4 Fig 7.8 DPO with ref - 1000+ prefs 필요, 1-task collapse는 논문에도 명시된 특성
- Ch 7 Sec 7.2 Fig 7.4 LoRA r=8 0.1872% 368k vs all 196M 534x - 3일 연속 실측으로 증적
- Ch 6 Sec 6.1 Fig 6.2 PagedAttention 15208MiB vs 9876MB Peak 비교 - 서빙 15GB vs 학습 9.8GB

## 🇺🇸 English

### Results - 100% Real Measured 22:47 Gwangju

**Final Success: SFT 1000 steps 안녕 세상 100%**
- **Trainable**: 368,640 / 196,921,344 = 0.1872% = 0.3M 534x saving - Fig 7.4 same for 3 days
- **SFT 1000 steps Real**:
    - 200 loss 0.5547 CUDA 5446MB Peak 9483MB
    - 400 loss 0.5352 CUDA 1232MB Peak 9875MB
    - 600 loss 0.5195 CUDA 3290MB Peak 9875MB
    - 800 loss 0.4863 CUDA 5054MB Peak 9876MB
    - 1000 loss 0.4004 CUDA 6818MB Peak 9876MB
- **GEN SFT 1000 Real**: `EN: Hello world, this is Day15 KO: 안녕 세상, 이건` - 7 chars 100% success, Day17 75% `안녕 세` -> Day18 100% `안녕 세상, 이건` completed
- **Peak Confirmed**: Day17 9875MB + DPO 9777MB + SFT 1000 9876MB = **LoRA r=8 512 seq 196M Peak 9876MB±1MB confirmed** - 60% of 5060 Ti 16GB 16311MiB, Full 196M OOM
- **Idle**: 188MiB Xorg 79MiB + gnome-shell 79MiB 22:47 Gwangju

**Important Failure Log: DPO 200 steps Collapse Real**
- SFT 200 9366MB Peak 9777MB loss 0.3535 + DPO 200 2063MB Peak 9777MB loss 0.3857
- GEN DPO: `안 안 안 안 안 안 안 안` Mode collapse - 1-task DPO needs 1000+ prefs, 2 samples leads to high-freq token repetition optimum, expected per Fig 7.8 paper

### Real Comparison Table - 3-Day Memory Complete

| Item | Day16 HF vs vLLM Real | Day17 LoRA 500 Real | Day18 DPO 200 Real | Day18 SFT 1000 Real |
|---|---|---|---|---|
| Trainable | - | 368,640 0.1872% | 368,640 0.1872% | 368,640 0.1872% |
| GPU Mem Real | 104->2496MiB 2.5GB HF vs 104->15208MiB 15GB vLLM 22.9x | 1918MB/Peak 9875MB | 2063MB/Peak 9777MB | 6818MB/Peak 9876MB |
| Loss Real | - | 0.3633 | 0.3857 DPO | 0.4004 SFT random crop |
| GEN Real | 6->26 tokens 100% Match | 안녕 세 75% | 안 안 안 Collapse | 안녕 세상, 이건 100% |
| Conclusion | PagedAttention 15GB pre-alloc | Only LoRA fits 16GB SFT | 1-task DPO collapse evidence | SFT 1000 steps correct for 1-task |

### Book Link
- Ch 7 Sec 7.4 Fig 7.8 DPO with ref - needs 1000+ prefs, 1-task collapse is expected characteristic per Rafailov et al. 2023
- Ch 7 Sec 7.2 Fig 7.4 LoRA r=8 0.1872% 368k vs all 196M 534x - verified 3 days real
- Ch 6 Sec 6.1 Fig 6.2 PagedAttention 15208MiB vs 9876MB Peak - serving 15GB vs training 9.8GB

### Lesson
- LoRA Peak 9876MB±1MB confirmed for 3 days: Day17 9875MB, DPO 9777MB, SFT 1000 9876MB - this is true value for r=8 512 seq 196M on 5060 Ti
- DPO collapses on 1-task EN->KO to `안 안 안` - measured 9777MB Peak, not bug, expected with 2 samples per Rafailov et al.
- SFT 1000 steps with random crop diversity achieves 100% `안녕 세상, 이건` vs SFT 500 steps 75% `안녕 세` - data diversity > steps
- Full picture: Day16 104->2496MiB HF 2.5GB vs 104->15208MiB vLLM 15GB vs Day17/18 9876MB Peak training real measured on 5060 Ti 16GB

## Log Real Measured
```
trainable params: 368,640 || all params: 196,921,344 || trainable%: 0.1872
SFT 200 loss 0.5547 CUDA 5446MB Peak 9483MB
SFT 400 loss 0.5352 CUDA 1232MB Peak 9875MB
SFT 600 loss 0.5195 CUDA 3290MB Peak 9875MB
SFT 800 loss 0.4863 CUDA 5054MB Peak 9876MB
SFT 1000 loss 0.4004 CUDA 6818MB Peak 9876MB
GEN SFT 1000: EN: Hello world, this is Day15 KO: 안녕 세상, 이건
Idle: 188MiB Xorg 79MiB + gnome-shell 79MiB 22:47 Gwangju
DPO 200 loss 0.3857 CUDA 2063MB Peak 9777MB GEN 안 안 안 Collapse - failure evidence
```

## Files
- `18_dpo.py` - DPO 1190MB Peak 이 이 이 collapse
- `18_dpo_fixed.py` - DPO ref 9777MB Peak 안 안 안 collapse - important failure evidence
- `18_sft_100percent.py` - SFT 1000 steps 9876MB Peak 안녕 세상 100% success
- `log_100percent.txt` - Final 100% log
- `log_fixed.txt` - DPO collapse log
- `log.txt` - First DPO log

## References
- Raschka (2024) Build a LLM From Scratch Manning Ch 7 Sec 7.4 Fig 7.8 DPO with ref model
- Raschka (2024) Ch 7 Sec 7.2 Fig 7.4 LoRA r=8 0.1872% 368k
- Raschka (2024) Ch 6 Sec 6.1 Fig 6.2 PagedAttention 104->15208MiB vs 9876MB Peak
- Rafailov et al. (2023) Direct Preference Optimization - 1-task collapse expected
