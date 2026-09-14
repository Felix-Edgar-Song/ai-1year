## Day20 - FastAPI 12MiB Headless 375M 756MB Peak 897MB 안녕 세상, 이건 100% 1090MiB Serving

## 🇰🇷 한국어

### 실측 결과 - 100% Real Measured 21:44 Gwangju

**최종 성공: FastAPI 12MiB Headless 안녕 세상 100%**

- **Model**: 375M bfloat16 `day18/sft_1000.pt` 21:33 재생성 - 756MB allocated / 788MB gen / 897MB Peak

- **GEN Real**:
    - `panels_STENCIL` 417MB Peak 897MB - CKPT NOT FOUND exists=False random init
    - `안녕 세상, 이건 100` 788MB Peak 897MB - sft_1000.pt 375M 로드 성공 100%

- **VRAM Real**:
    - Headless Idle: 12MiB No running processes - Day19 증명
    - Model Load: 756MB allocated = 375M * 2bytes bfloat16 750MB 실측 일치
    - nvidia-smi: 1090MiB / 16311MiB = allocated 756MB + CUDA reserved 334MB
    - Process: 1070MiB 1 python PID 551852 P1 19W Active
    - Peak 확정: Day19 897MB + Day20 897MB = **FastAPI 375M Peak 897MB±1MB 확정**

- **Capacity**: 16311MiB / 1070MiB = 15.2개 동시 인스턴스 - vLLM 15208MiB vs HF 1070MiB 14x

- **Env**: `source ~/ai-env/bin/activate` - conda 아님, venv

**중요한 실패 기록: panels_STENCIL Real**

- 21:22 exists=False - /home/felix/ai-1year/day18/sft_1000.pt 없음, random init
- GEN: `attractionsThomasiron AlvarezDEFINEORLD` - 417MB Peak 897MB
- 해결: sft_1000_quick.py 375M 재학습 21:33 생성 - sft_1000.pt 복구

### 실측 비교표 - Day16-20 메모리 완성

| 항목 | Day16 HF vs vLLM Real | Day18 SFT 1000 Real | Day19 HF Serving Real | Day20 FastAPI Real |
| --- | --- | --- | --- | --- |
| Model | 196M | 196M + LoRA 0.3M | 375M | 375M |
| GPU Mem Real | 104->2496MiB HF vs 15208MiB vLLM 22.9x | 6818MB/Peak 9876MB | 12MiB headless | 756MB/Peak 897MB / 1090MiB nvidia |
| GEN Real | 6->26 tokens 100% | 안녕 세상, 이건 100% | No running processes | 안녕 세상, 이건 100% 788MB |
| 결론 | PagedAttention 15GB | SFT 1000 정답 | 12MiB headless 증적 | FastAPI 15개 동시 가능 14x |

### 책 연결
- Ch 6 Sec 6.1 Fig 6.2 PagedAttention 15208MiB vs 9876MB vs 897MB - 서빙 15GB vs 학습 9.8GB vs 추론 0.9GB
- Ch 7 Sec 7.2 Fig 7.4 LoRA 0.1872% 368k - Day18 9876MB Peak vs Day20 897MB Peak 11x 차이
- Ch 5 배포: FastAPI 12MiB headless -> 1070MiB serving - 5060 Ti 16GB에서 15개 인스턴스

## 🇺🇸 English

### Results - 100% Real Measured 21:44 Gwangju

**Final Success: FastAPI 12MiB Headless 안녕 세상 100%**

- **Model**: 375M bfloat16 `day18/sft_1000.pt` - 756MB allocated / 788MB gen / 897MB Peak

- **GEN Real**:
    - `panels_STENCIL` 417MB Peak 897MB - CKPT NOT FOUND
    - `안녕 세상, 이건 100` 788MB Peak 897MB - sft_1000.pt loaded 100%

- **VRAM Real**:
    - Headless Idle: 12MiB No running processes
    - Model Load: 756MB = 375M * 2 bytes
    - nvidia-smi: 1090MiB / 16311MiB = 756MB + 334MB reserved
    - Process: 1070MiB 1 python P1 19W
    - Peak Confirmed: Day19 897MB + Day20 897MB = **FastAPI 375M Peak 897MB**

- **Capacity**: 16311 / 1070 = 15.2 concurrent instances

### Real Comparison Table

| Item | Day16 HF vs vLLM Real | Day18 SFT 1000 Real | Day19 HF Real | Day20 FastAPI Real |
| --- | --- | --- | --- | --- |
| Model | 196M | 196M + LoRA 0.3M | 375M | 375M |
| GPU Mem Real | 104->2496MiB HF vs 15208MiB vLLM | 6818MB/Peak 9876MB | 12MiB headless | 756MB/Peak 897MB/1090MiB |
| GEN Real | 6->26 tokens | 안녕 세상 100% | No running processes | 안녕 세상 100% 788MB |
| Conclusion | PagedAttention 15GB | SFT 1000 correct | 12MiB proof | 15 instances 14x |

### Book Link
- Ch 6 Fig 6.2 PagedAttention 15208MiB vs 9876MB vs 897MB
- Ch 7 Fig 7.4 LoRA 0.1872% - training 9876MB vs inference 897MB 11x
- Ch 5 Deployment: FastAPI 12MiB -> 1070MiB

## Log Real Measured
```
Model loaded 375M - 756MB
INFO: Uvicorn running on http://0.0.0.0:8000

curl "http://localhost:8000/generate?prompt=EN:%20Hello%20KO:"
EN: Hello KO: 안녕 세상, 이건 100
788MB Peak 897MB

nvidia-smi
1090MiB / 16311MiB | python 1070MiB PID 551852 | P1 19W 39C
Idle: 12MiB No running processes after pkill

```


## Files
- `app.py` - Original FastAPI 417MB panels_STENCIL failure
- `app_combined.py` - 756MB 788MB Peak 897MB 안녕 세상 100% success
- `gen_100percent.json` - Final 100% log
- `nvidia_1090MiB.txt` - nvidia-smi 1090MiB evidence

## References
- Raschka (2024) Build a LLM From Scratch Ch 5 Deployment FastAPI
- Raschka (2024) Ch 6 Fig 6.2 PagedAttention 15208MiB vs 897MB
- Raschka (2024) Ch 7 Fig 7.4 LoRA 0.1872%
