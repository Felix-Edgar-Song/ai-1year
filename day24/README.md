# Day24 - Blackwell sm_120 Native Proof

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **1개 컨테이너** | `922MiB / 902MiB P8 5W` `health 377MB` | Day23 922MiB 동일, 760MiB 목표 미달 - Blackwell 최소 한계 |
| **17개 컨테이너** | `15438MiB / 16311MiB 94.6%` `902MiB x17` `P8 5W 37C` | 21개 시도 중 17개 성공, Day23과 동일 94.6% 한계 |
| **이론 vs 실측** | `16311 / 902 = 18.0개 이론` → `17개 실측` | 18번째부터 CUDA OOM, Day23과 동일 패턴 |
| **정리 후** | `12MiB P8 5W No running processes` | Day19/21/22/23과 동일하게 완전 복귀 |

**왜 760MiB가 아니라 902MiB인가?**

- Day23 `pytorch:2.4.0-cuda12.4-cudnn9-runtime`: 902MiB sm_90 경고 있음
- Day24 `python:3.12-slim + torch cu128`: 902MiB sm_120 네이티브, 경고 없음
- `python:3.12-slim`과 `pytorch-runtime`이 Blackwell에서는 동일한 CUDA context 902MiB가 최소 비용
- 모델 377MB + CUDA context 525MB = 902MiB가 Blackwell 하한선

**증명된 것:**

- `sm_120 is not compatible` 경고 완전 제거 - cu128 wheel이 Blackwell 네이티브 지원
- `ERROR fastapi not found` → `pip install fastapi && pip install torch --index-url cu128` 분리로 해결
- `Failed to initialize NumPy` → `pip install numpy` 추가로 해결
- `8000-8020:8000` 21개 중 `8001~8017` 17개 `Up`, `8000,8018-8020` 4개 OOM
- `nvidia-smi` 15438MiB는 5060 Ti 16GB의 94.6% 실사용 한계, Day23과 100% 동일
- Docker는 `ai-env`와 별개 - 시스템 레벨 `/var/run/docker.sock`

---

## 🇺🇸 English

### Measured Results

| Metric | Measured Value | Meaning |
| --- | --- | --- |
| **1 Container** | `922MiB / 902MiB P8 5W` `health 377MB` | Same as Day23 922MiB, 760MiB target missed - Blackwell minimal limit |
| **17 Containers** | `15438MiB / 16311MiB 94.6%` `902MiB x17` `P8 5W 37C` | 17 out of 21 succeeded, same 94.6% limit as Day23 |
| **Theory vs Reality** | `16311 / 902 = 18.0 theoretical` → `17 actual` | 18th OOM, same pattern as Day23 |
| **After cleanup** | `12MiB P8 5W No running processes` | Full return, same as Day19/21/22/23 |

**Why 902MiB not 760MiB?**

- Day23 `pytorch:2.4.0-cuda12.4-cudnn9-runtime`: 902MiB with sm_90 warning
- Day24 `python:3.12-slim + torch cu128`: 902MiB sm_120 native, no warning
- Both bases have same 902MiB CUDA context cost on Blackwell - minimal
- Model 377MB + CUDA context 525MB = 902MiB Blackwell floor

**Proven:**

- `sm_120 is not compatible` fully fixed - cu128 wheel natively supports Blackwell
- `ERROR fastapi not found` → split `pip install fastapi && pip install torch --index-url cu128`
- `Failed to initialize NumPy` → added `pip install numpy`
- Of `8000-8020:8000` 21 ports, 17 `Up` (8001~8017), 4 OOM (8000,8018-8020)
- `nvidia-smi` 15438MiB is 94.6% real limit of 5060 Ti 16GB, identical to Day23
- Docker is separate from `ai-env` - system-level `/var/run/docker.sock`

---

## 📊 Logs / 로그
```
1x
nvidia_760MiB_1x.txt # target 760MiB, actual 922MiB / 902MiB python / health 377MB

21x attempt / 21개 시도
nvidia_16000MiB_21x.txt # 17 containers 15438MiB 94.6% 902MiB x17
docker_ps_21x.txt # 17 Up, 4 OOM
count_21x.txt # 17
health_21x.txt # 8001∼8017 17 OK 377MB, 8000,8018-8020 Fail
nvidia_12Mib_return.txt # 12MiB P8 5W return / 복귀
```


## 🔧 Stack / 스택

- Version / 버전: Docker CE 28.x, Compose v2, NVIDIA Container Toolkit 1.16.x, CUDA 13.2, Driver 595.84
- Base Image / 베이스 이미지: python:3.12-slim + torch cu128 (sm_120 native) → python 902MiB
- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell
- Model / 모델: 375M sft_1000.pt 757MB bfloat16, gen 413MB Peak 417MB health 377MB

## 💡 Next / 다음

Day25: vLLM PagedAttention / SGLang 도입으로 902MiB → 450MiB → 30 containers 도전
Day25: vLLM to reduce 902MiB → 450MiB → 30 containers challenge
