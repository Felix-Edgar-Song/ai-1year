# Day23 - Docker Multi-Container Lightweight Proof

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **1개 컨테이너** | `922MiB / 902MiB P8 5W` `health 380MB` | Day22 1090MiB 대비 -168MiB 경량화, ai-env 766MiB 대비 +136MiB |
| **17개 컨테이너** | `15438MiB / 16311MiB 94.6%` `902MiB x17` `P8 5W 37C` | 21개 시도 중 17개 성공, Day22 14개 대비 +3개 |
| **이론 vs 실측** | `16311 / 902 = 18.0개 이론` → `17개 실측` | 18번째부터 CUDA OOM Restarting |
| **정리 후** | `12MiB P8 5W No running processes` | Day19/21/22와 동일하게 완전 복귀 |

**왜 902MiB인가?**

- Day22 `python:3.12-slim`: `torch 2.14 554MB + cudnn 553MB` = 1070MiB
- Day23 `pytorch:2.4.0-cuda12.4-cudnn9-runtime`: base 경량화로 902MiB
- 차이 `168MiB` 감소가 14→17개 돌파 원인

**증명된 것:**

- Docker는 `ai-env`와 별개 - 시스템 레벨 `/var/run/docker.sock`
- `build:.` → `build: .` 스페이스 1칸 mapping error 해결
- `COPY app_v2.py.` → `COPY app_v2.py .` 스페이스 1칸 COPY args 해결
- `CKPT /home/felix/ai-1year/day18/sft_1000.pt` 볼륨 마운트로 AssertionError 해결
- `deploy.resources` → `runtime: nvidia`로 `nvidia-cuda-mps-control` not found 해결
- `8000-8020:8000` 21개 포트 중 `8000~8016` 17개 `Up`, 4개 Restarting OOM
- `nvidia-smi` 15438MiB는 5060 Ti 16GB의 94.6% 실사용 한계

---

## 🇺🇸 English

### Measured Results

| Metric | Measured Value | Meaning |
| --- | --- | --- |
| **1 Container** | `922MiB / 902MiB P8 5W` `health 380MB` | -168MiB vs Day22 1090MiB, +136MiB vs native ai-env 766MiB |
| **17 Containers** | `15438MiB / 16311MiB 94.6%` `902MiB x17` `P8 5W 37C` | 17 out of 21 succeeded, +3 vs Day22 14 |
| **Theory vs Reality** | `16311 / 902 = 18.0 theoretical` → `17 actual` | 18th container OOM Restarting |
| **After cleanup** | `12MiB P8 5W No running processes` | Full return, same as Day19/21/22 |

**Why 902MiB?**

- Day22 `python:3.12-slim`: `torch 2.14 554MB + cudnn 553MB` = 1070MiB
- Day23 `pytorch:2.4.0-cuda12.4-cudnn9-runtime`: lightweight base = 902MiB
- `168MiB` reduction enables 14→17 breakthrough

**Proven:**

- Docker is separate from `ai-env` - system-level `/var/run/docker.sock`
- `build:.` → `build: .` space fix mapping error
- `COPY app_v2.py.` → `COPY app_v2.py .` space fix COPY args error
- `CKPT /home/felix/ai-1year/day18/sft_1000.pt` volume mount fix AssertionError
- `deploy.resources` → `runtime: nvidia` fix `nvidia-cuda-mps-control` not found
- Of `8000-8020:8000` 21 ports, 17 `Up` (8000~8016), 4 Restarting OOM
- `nvidia-smi` 15438MiB is 94.6% real limit of 5060 Ti 16GB

---

## 📊 Logs / 로그
```
1x
nvidia_760MiB_1x.txt # Day23 target 760MiB, actual 922MiB
nvidia_760MiB_1x.txt content: 922MiB / 902MiB python

21x attempt / 21개 시도
nvidia_16000MiB_21x.txt # 17 containers 15438MiB 94.6% 902MiB x17
docker_ps_21x.txt # 21 created, 17 Up, 4 Restarting
health_21x.txt # 17x health check
oom_4x.txt # 4 Restarting (api-6,7,8,19)

Cleanup / 정리 후
nvidia_12MiB_return.txt # 12MiB P8 5W return / 복귀
nvidia_12MiB_after_21x.txt # same as above
```


## 🔧 Stack / 스택

- Version / 버전: Docker CE 28.x, Compose v2, NVIDIA Container Toolkit 1.16.x, CUDA 13.2, Driver 595.84

- Base Image / 베이스 이미지: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime → python 902MiB

- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell

- Model / 모델: 375M sft_1000.pt 757MB bfloat16, gen 413MB Peak 417MB

## 💡 Next / 다음

- Day24: python:3.12-slim + torch 2.8.0 cu128 (sm_120 native) → 760MiB target → 21 containers full Up challenge

- Day24: cu128 base for sm_120 support to achieve 760MiB → 21 containers full challenge
