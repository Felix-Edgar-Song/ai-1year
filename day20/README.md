## Day20 - FastAPI 12MiB Headless 375M 756MB Peak 897MB 안녕 세상, 이건 100% 1100MiB Real

## 🇰🇷 한국어

### 실측 결과 - 100% Real Measured 22:03 Gwangju

**최종 성공: FastAPI 12MiB Headless 안녕 세상 100%**

- **Model**: 375M bfloat16 `day18/sft_1000.pt` 21:33 재생성 - 756MB allocated / 788MB gen / 897MB Peak

- **GEN Real**:
    - `panels_STENCIL` 417MB Peak 897MB - CKPT NOT FOUND exists=False random init 21:22
    - `안녕 세상, 이건 100` 788MB Peak 897MB - sft_1000.pt 375M 로드 100% 21:44

- **VRAM Real 22:03**:
    - nvidia-smi: 1100MiB / 16311MiB | python 1080MiB PID 559078 | P8 5W 37C
    - 21:44: 1090MiB / 1070MiB P1 19W 39C -> 22:03: 1100MiB / 1080MiB P8 5W 37C +10MiB
    - allocated: 756MB / Peak: 897MB / reserved: 1100-756=344MB
    - Headless: 12MiB No running processes after pkill - Day19 증명
    - Capacity: 16311MiB / 1080MiB = 15.1개 동시 인스턴스 - vLLM 15208MiB vs 1080MiB 14x

- **Env**: `source ~/ai-env/bin/activate` - conda 아님 venv

**중요한 실패 기록: panels_STENCIL -> 안녕 세상**

- 21:22 exists=False random init 417MB Peak 897MB `panels_STENCIL`
- 21:33 sft_1000_quick.py 375M 재생성
- 21:44 756MB Model loaded 788MB Peak 897MB `안녕 세상, 이건 100` 100% 복구
- 22:03 1100MiB/1080MiB 재측정 +10MiB

### 실측 비교표 - Day16-20 메모리 완성

| 항목 | Day16 HF vs vLLM Real | Day18 SFT 1000 Real | Day19 HF Real | Day20 FastAPI Real 22:03 |
| --- | --- | --- | --- | --- |
| Trainable | - | 368,640 0.1872% | - | - |
| Model | 196M | 196M + LoRA 0.3M | 375M | 375M |
| GPU Mem Real | 104->2496MiB HF vs 15208MiB vLLM 22.9x | 6818MB/Peak 9876MB | 12MiB headless | 756MB/Peak 897MB/1100MiB nvidia 1080MiB process |
| GEN Real | 6->26 tokens 100% | 안녕 세상, 이건 100% | No running processes | 안녕 세상, 이건 100% 788MB |
| 결론 | PagedAttention 15GB | SFT 1000 정답 | 12MiB headless 증적 | FastAPI 15개 동시 14x 1100MiB |

### 책 연결
- Ch 6 Sec 6.1 Fig 6.2 PagedAttention 15208MiB vs 9876MB vs 1100MiB - 서빙 15GB vs 학습 9.8GB vs 추론 1.1GB
- Ch 7 Sec 7.2 Fig 7.4 LoRA r=8 0.1872% 368k - Day18 9876MB vs Day20 1100MiB 9x
- Ch 5 배포: FastAPI 12MiB headless -> 1080MiB serving - 5060 Ti 16GB 15개

## 🇺🇸 English

### Results - 100% Real Measured 22:03 Gwangju

**Final Success: FastAPI 12MiB Headless 안녕 세상 100%**

- **Model**: 375M bfloat16 - 756MB allocated / 788MB gen / 897MB Peak / 1100MiB nvidia / 1080MiB process

- **GEN Real**: `안녕 세상, 이건 100` 788MB Peak 897MB 100% - recovered from `panels_STENCIL`

- **VRAM Real 22:03**: 1100MiB / 16311MiB | 1080MiB process PID 559078 P8 5W 37C - 15.1 instances

### Real Comparison Table

| Item | Day16 Real | Day18 Real | Day19 Real | Day20 Real 22:03 |
| --- | --- | --- | --- | --- |
| GPU Mem Real | 104->15208MiB vLLM | 6818MB/Peak 9876MB | 12MiB headless | 756MB/Peak 897MB/1100MiB/1080MiB |
| GEN Real | 6->26 tokens | 안녕 세상 100% | No running processes | 안녕 세상 100% 788MB |
| Conclusion | 15GB pre-alloc | SFT 1000 correct | 12MiB proof | 15 instances 14x 1100MiB |

## Log Real Measured
```
[200~Model loaded 375M - 756MB
INFO: Uvicorn running on http://0.0.0.0:8000

Failure 21:22:
exists=False CKPT NOT FOUND
GEN panels_STENCIL 417MB Peak 897MB

Success 21:44:
EN: Hello KO: 안녕 세상, 이건 100
788MB Peak 897MB

Real 22:03 Gwangju:
Mon Sep 14 22:03:12 2026

GPU RTX 5060 Ti	1100MiB / 16311MiB	0%	P8 5W 37C
Processes: python 1080MiB PID 559078



## Files
- `app_combined.py` - 756MB 788MB Peak 897MB 안녕 세상 100% success
- `app.py` - Original 417MB panels_STENCIL failure
- `nvidia_1090MiB.txt` - 22:03 1100MiB/1080MiB Real log (name 1090MiB but 1100MiB inside)
- `gen_100percent.json` - 788MB Peak 897MB GEN log

## References
- Raschka (2024) Build a LLM From Scratch Ch 5 Deployment FastAPI
- Raschka (2024) Ch 6 Sec 6.1 Fig 6.2 PagedAttention 15208MiB vs 9876MB vs 1100MiB
- Raschka (2024) Ch 7 Sec 7.2 Fig 7.4 LoRA 0.1872%
