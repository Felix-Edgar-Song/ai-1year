# Day22 - Docker Multi-Container Capacity Proof

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
|---|---|---|
| **1개 컨테이너** | `1090MiB / 1070MiB P1 18W` `health 380MB` | ai-env 네이티브 766MiB 대비 +304MiB 오버헤드 |
| **14개 컨테이너** | `15068MiB / 16311MiB 92.4%` `1070MiB x14` `P8 5W 36C` | 21개 시도 중 14개만 성공, 92.4%에서 OOM |
| **이론 vs 실측** | `16311 / 1070 = 15.2개 이론` → `14개 실측` | 15번째 컨테이너부터 CUDA OOM으로 exit |
| **정리 후** | `12MiB P8 5W No running processes` | Day19/21과 동일하게 완전 복귀 |

**왜 1070MiB인가?**
- 네이티브 `ai-env`: `torch` 1개, 766MiB (모델 757MB + CUDA context 9MB)
- Docker `python:3.12-slim`: `torch 2.14 554MB + cudnn 553MB = 1.1GB` base 이미지 오버헤드
- 차이 `304MiB`가 Docker base 비용

**증명된 것:**
- Docker는 `ai-env`와 별개 - 시스템 레벨 `/var/run/docker.sock`
- `8000-8020:8000` 21개 포트 중 `8000~8013` 14개만 `Up`, 나머지는 메모리 부족으로 죽음
- `nvidia-smi` 15068MiB는 5060 Ti 16GB의 실제 한계

---

## 🇺🇸 English

### Measured Results

| Metric | Measured Value | Meaning |
|---|---|---|
| **1 Container** | `1090MiB / 1070MiB P1 18W` `health 380MB` | +304MiB overhead vs native ai-env 766MiB |
| **14 Containers** | `15068MiB / 16311MiB 92.4%` `1070MiB x14` `P8 5W 36C` | 14 out of 21 succeeded, OOM at 92.4% |
| **Theory vs Reality** | `16311 / 1070 = 15.2 theoretical` → `14 actual` | 15th container exits with CUDA OOM |
| **After cleanup** | `12MiB P8 5W No running processes` | Full return, same as Day19/21 |

**Why 1070MiB?**
- Native `ai-env`: single `torch`, 766MiB (model 757MB + CUDA context 9MB)
- Docker `python:3.12-slim`: `torch 2.14 554MB + cudnn 553MB = 1.1GB` base image overhead
- `304MiB` difference is Docker base cost

**Proven:**
- Docker is separate from `ai-env` - system-level `/var/run/docker.sock`
- Of `8000-8020:8000` 21 ports, only `8000~8013` 14 `Up`, rest died from OOM
- `nvidia-smi` 15068MiB is the real limit of 5060 Ti 16GB

---

## 📊 Logs / 로그

```bash
# 1x
nvidia_1090MiB_1x.txt        # 1 container 1090MiB
# 14x
nvidia_15068MiB_14x.txt      # 14 containers 15068MiB 92.4%
docker_ps_14x.txt            # 14 containers Up
health_14x.txt               # 14x health 380MB
# Cleanup / 정리 후
nvidia_12MiB_after_14x.txt   # 12MiB P8 5W return / 복귀
```

🔧 Stack / 스택
Version / 버전: Docker CE 28.x, Compose v2, NVIDIA Container Toolkit 1.16.x, CUDA 13.2, Driver 595.84
Base Image / 베이스 이미지: python:3.12-slim + torch 2.14.0 554MB + cudnn 553MB
GPU: RTX 5060 Ti 16GB 16311MiB
Model / 모델: 375M sft_1000.pt 757MB bfloat16, gen 413MB Peak 417MB 안녕 세상 100%
💡 Next / 다음
Day23: pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime base로 1070MiB → 800MiB 경량화 → 20개 도전
Day23: Lightweight base to reduce 1070MiB → 800MiB → 20 containers challenge
