# Day25 - vLLM PagedAttention 450MiB Challenge

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **초기 1개** | `5818MiB / 5838MiB P1 19W` `gpu_util 0.35 len 512` | CUDA 12.9 + FlashInfer 업데이트 전, KV cache 4.82GiB 폭주 |
| **최적화 1개** | `1418MiB / 1398MiB P8 5W` `gpu_util 0.15 len 128` | KV cache 0.85GiB로 75% 절감 |
| **초저용량 1개** | `628MiB / 608MiB P8 5W` `gpu_util 0.08 len 128 seq1` | 89.5% 절감, vLLM 오버헤드 포함 최소치 |
| **3개 컨테이너** | `2128MiB / 16311MiB 13%` `552 / 686 / 860MiB P8 5W 37C` | 8000,8001,8004 3개 Up, 평균 700MiB, 16GB 대비 13% |
| **5개 시도** | `2128MiB 3 Up, 2 OOM` `Available KV -0.28GiB` | gpu_util 0.08에서는 4번째부터 No available memory, fragmentation |
| **이론 vs 실측** | `16311 / 608 = 26.8개 이론` → `3개 실측 (0.08)` `10개 실측 (0.15)` | 0.08은 메모리 파편화로 10개 이상 불안정, 0.15가 8~10개 안정적 |
| **정리 후** | `12MiB P8 5W No running processes` | Day24와 동일하게 완전 복귀 |

**왜 450MiB가 아니라 608MiB인가?**

- Day24 `python:3.12-slim + torch cu128`: 902MiB (모델 377MB + CUDA context 525MB)
- Day25 `vLLM gpt2 124M bfloat16`: 608MiB (가중치 248MiB + EngineCore 360MiB)
- vLLM 자체 오버헤드 (NCCL + EngineCore + CUDA context)가 최소 360MiB 고정
- KV cache를 `max_model_len=128, max_num_seqs=1, gpu_util=0.08`로 최소화해도 608MiB가 Blackwell + vLLM 하한선
- 모델 248MiB + 엔진 360MiB = 608MiB, 450MiB로 가려면 vLLM이 아닌 `torch.compile` 필요

**증명된 것:**

- `SM 12.x requires CUDA >= 12.9` → `nvidia/cuda:12.9.0-devel` 교체로 해결, Blackwell 네이티브
- `FlashInfer requires sm75+` → `pip install -U flashinfer-python` + `VLLM_USE_FLASHINFER_SAMPLER=0`로 해결
- `Failed to build Triton: No C compiler` → `python:3.12-slim`에 `build-essential python3-dev` 추가로 해결
- `no space left on device` → CUDA 12.9 devel 8GB + torch 3GB + vllm 2GB = 15GB+, `docker system prune -a`로 해결
- `Bind 0.0.0.0:8000 failed: port allocated` → `--scale api=5`가 같은 호스트 포트 바인딩, `8000-8004` 분리로 해결
- `No available memory for cache blocks -0.28GiB` → gpu_util 0.08에서 4개 이상 동시 기동 시 OOM, 0.15로 올려야 8개 이상 가능
- `nvidia-smi` 2128MiB는 3개 vLLM 엔진의 실제 합계, 단일 엔진은 194 concurrent 처리 가능 (continuous batching)

### 💡 핵심 포인트

- **450MiB 실패가 아닌 608MiB 한계 증명:** Day24 PyTorch 902MiB → Day25 vLLM 608MiB로 32% 추가 절감, vLLM PagedAttention이 902MiB 한계를 깼음 - 하지만 450MiB는 vLLM 구조상 불가능, 순수 torch가 필요
- **Blackwell sm_120 vLLM 네이티브 달성:** Day24 `sm_90 warning` → Day25 `Using FLASH_ATTN backend, FlashAttention v2` 경고 0개, `VLLM_USE_FLASHINFER_SAMPLER=0`으로 우회 성공
- **3개가 아닌 1개 배칭이 정답:** 3개 컨테이너 2128MiB vs 1개 컨테이너 608MiB로 30개 요청을 continuous batching으로 처리, 복제보다 배칭이 30배 효율 - 16GB에서 26개 이론상 가능하지만 1개가 더 효율적
- **QnA:** "왜 450MiB 안됐나요? vLLM EngineCore 자체가 360MiB 고정 오버헤드 + 가중치 248MiB = 608MiB가 하한선, 450MiB로 가려면 vLLM이 아닌 torch.compile + bfloat16으로 가야 합니다" 라고 답하면 됨

---

## 🇺🇸 English

### Measured Results

| Metric | Measured Value | Meaning |
| --- | --- | --- |
| **Initial 1x** | `5818MiB / 5838MiB P1 19W` `gpu_util 0.35 len 512` | Before CUDA 12.9 + FlashInfer update, KV cache 4.82GiB explosion |
| **Optimized 1x** | `1418MiB / 1398MiB P8 5W` `gpu_util 0.15 len 128` | KV cache 0.85GiB, 75% reduction |
| **Ultra-low 1x** | `628MiB / 608MiB P8 5W` `gpu_util 0.08 len 128 seq1` | 89.5% reduction, minimal with vLLM overhead |
| **3 Containers** | `2128MiB / 16311MiB 13%` `552 / 686 / 860MiB P8 5W 37C` | 3 Up (8000,8001,8004), avg 700MiB, 13% of 16GB |
| **5 Attempt** | `2128MiB 3 Up, 2 OOM` `Available KV -0.28GiB` | OOM from 4th with gpu_util 0.08, fragmentation |
| **Theory vs Reality** | `16311 / 608 = 26.8 theoretical` → `3 actual (0.08)` `10 actual (0.15)` | 0.08 unstable over 10 due to fragmentation, 0.15 stable 8-10 |
| **After cleanup** | `12MiB P8 5W No running processes` | Full return, same as Day24 |

**Why 608MiB not 450MiB?**

- Day24 `python:3.12-slim + torch cu128`: 902MiB (model 377MB + CUDA context 525MB)
- Day25 `vLLM gpt2 124M bfloat16`: 608MiB (weights 248MiB + EngineCore 360MiB)
- vLLM fixed overhead (NCCL + EngineCore + CUDA context) min 360MiB
- Even with `max_model_len=128, max_num_seqs=1, gpu_util=0.08`, 608MiB is floor for Blackwell + vLLM
- Model 248MiB + engine 360MiB = 608MiB, need `torch.compile` for 450MiB, not vLLM

**Proven:**

- `SM 12.x requires CUDA >= 12.9` → fixed by `nvidia/cuda:12.9.0-devel`, Blackwell native
- `FlashInfer requires sm75+` → fixed by `pip install -U flashinfer-python` + `VLLM_USE_FLASHINFER_SAMPLER=0`
- `Failed to build Triton: No C compiler` → fixed by `build-essential python3-dev` on slim
- `no space left on device` → CUDA devel 8GB + torch 3GB + vllm 2GB = 15GB+, fixed by `docker system prune -a`
- `Bind 0.0.0.0:8000 failed: port allocated` → `--scale api=5` same host port, fixed by splitting `8000-8004`
- `No available memory for cache blocks -0.28GiB` → OOM from 4th with 0.08, need 0.15 for 8+
- `nvidia-smi` 2128MiB is sum of 3 vLLM engines, single engine handles 194 concurrent via continuous batching

### 💡 Key Insight

- **Not failure but proof of 608MiB floor:** Day24 PyTorch 902MiB → Day25 vLLM 608MiB, 32% extra reduction, PagedAttention broke 902MiB wall - but 450MiB impossible with vLLM structure, need vanilla torch
- **Blackwell sm_120 vLLM native achieved:** Day24 `sm_90 warning` → Day25 `Using FLASH_ATTN backend` 0 warnings, bypassed with `VLLM_USE_FLASHINFER_SAMPLER=0`
- **Batching beats replication:** 3 containers 2128MiB vs 1 container 608MiB handling 30 requests via batching, 30x more efficient - 26 theoretical but 1 is better
- **QnA:** "Why not 450MiB? vLLM EngineCore fixed 360MiB overhead + weights 248MiB = 608MiB floor. For 450MiB, need torch.compile + bfloat16, not vLLM"

---

## 📊 Logs / 로그
```
1x initial
nvidia_5818MiB_1x.txt # gpu_util 0.35 len 512, 5838MiB P1 19W, KV 4.82GiB

1x optimized
nvidia_1418MiB_1x.txt # gpu_util 0.15 len 128, 1418MiB 1398MiB P8 5W, KV 0.85GiB, concurrency 194x
nvidia_608MiB_1x.txt # gpu_util 0.08 len 128 seq1, 628MiB 608MiB P8 5W, final single best

3x concurrent / 3개 동시
nvidia_2128MiB_3x.txt # 3 containers 2128MiB 13% 552/686/860MiB P8 5W 37C
docker_ps_3x.txt # 3 Up (8000,8001,8004)
generate_3x.txt # 8000,8001,8004 all OK Snowbat woke up...

5x attempt OOM / 5개 시도 OOM
nvidia_0.28GiB_OOM.txt # Available KV -0.28GiB No available memory
docker_ps_5x.txt # 5 created, 3 Up, 2 Exit OOM

cleanup / 정리
nvidia_12MiB_return.txt # 12MiB P8 5W No running processes return / 복귀
```


## 🔧 Stack / 스택

- Version / 버전: Docker CE 28.x, Compose v2, NVIDIA Container Toolkit 1.16.x, CUDA 13.2, Driver 595.84
- Base Image / 베이스 이미지: nvidia/cuda:12.9.0-devel-ubuntu22.04 + torch cu128 + flashinfer + vllm 0.29.0 → 608MiB ultra-low
- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell
- Model / 모델: openai-community/gpt2 124M bfloat16 0.24GiB weights, Peak 0.6GiB, health OK
- vLLM Config / 설정: gpu_memory_utilization=0.08, max_model_len=128, max_num_seqs=1, enforce_eager=True, prefix_caching=False

## 💡 Next / 다음

- Day26: torch.compile + bfloat16으로 608MiB → 400MiB 도전, 375M CausalGPTCustom 902MiB → 400MiB
- Day26: torch.compile to reduce 608MiB → 400MiB, 375M model 902MiB → 400MiB challenge
- Day27: SGLang / TensorRT-LLM Blackwell 네이티브 테스트, 30 containers vs 1 engine batching 비교
- Day27: SGLang / TensorRT-LLM Blackwell native test, compare 30 containers vs 1 engine batching
