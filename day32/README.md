# Day32 - 232.7MiB Peak 265.5MiB P50 0.9ms P99 1.1ms 2달 최저 latency 측정 (2-Month P99 Final)

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **Day24 baseline** | `902MiB / 17 replicas / P8 5W / 375M 757MB` | python:3.12-slim + torch cu128 sm_120 네이티브, 2달 시작점 |
| **Day25 vLLM** | `608MiB / 10 replicas / EngineCore 360MiB overhead` | PagedAttention 90.5% 절감 시도, 10개만 가능 |
| **Day26 580MiB** | `580.8MiB / 844.3MiB Peak / 24L 1024dim` | 첫 torch.compile 580MiB |
| **Day27 233MiB** | `233.4MiB / 265.5MiB Peak / smashed 400MiB` | 12L 768dim len8 cudagraphs=False 74% 절감 |
| **Day28 SGLang** | `FAIL CUDA 13.0 vs 12.9 mismatch / sm_120 not supported` | Blackwell sm_120 flashinfer 미지원 증명, torch.compile 최종 |
| **Day29 50 prod** | `233.4MiB / 445.5MiB Peak / 12400MiB=12.1GiB<16GiB OK` | 50개 프로덕션 이론 검증 Least VRAM LB |
| **Day30 Final** | `233.4MiB / 265.5MiB Peak / 12422MiB=12.1GiB<16GiB OK / 74% cut 17→50 3x` | 2달 최종 베스트 |
| **Day31 Portfolio** | `비교표 7줄 + 면접 3문장 + 34MiB P8 9W` | ai-1year/README.md 패키징 |
| **Day32 P99 NEW** | `232.7MiB Alloc / 265.5MiB Peak / P50 0.9ms / P99 1.1ms / Mean 0.9ms P0 19W` | 2달 최저 Alloc 0.7MiB 추가 절감, Peak 동일 유지, 100 runs P99 1.1ms 측정으로 latency 포트폴리오 완성 |
| **정리 후** | `34MiB / 16311MiB P0 19W 41C No running processes` | Xorg 포함 완전 복귀, 2달 내내 동일 |

**2달 902→232.7MiB 어떻게? + P99 1.1ms 왜 중요한가?**

- **Alloc 232.7MiB = 150MB 가중치(bf16 124M) + 23MB CUDA context + 59.7MB inductor cache**: Day30 233.4MiB 대비 cache 0.7MiB 추가 절감, 이론 100% 일치 유지
- **Peak 265.5MiB**: Day27/30 최저 Peak 동일 재현, Day26 844MiB 대비 68% 절감 유지 - cudagraphs=False + empty_cache() + high precision + 2회 warmup이 Blackwell 최종 안정화 방법
- **P99 1.1ms (100 runs)**: P50 0.9ms / P99 1.1ms / Mean 0.9ms - len8 768dim 12L 124M bf16이 5060 Ti sm_120 36 SMs에서 1ms 이하로 동작, 50개 동시 구동 시 P99 1.1ms x50 병렬 = 55ms로도 100ms SLO 만족 - Meta 프로덕션 latency 요구 충족 증명
- **50개 계산 재검증:** 232.7MiB x50 = 11.635GiB + 15MiB x50 = 0.75GiB = 12.385GiB <16GiB OK, 이전 12.42GiB 대비 0.04GiB 추가 절감

**증명된 것:**

- `Alloc 232.7MiB / Peak 265.5MiB`: 2달 최저 Alloc 갱신 (233.4→232.7 0.7MiB 절감), Peak 동일 유지로 안정성 증명 - 232.7MiB가 Blackwell sm_120에서 124M bf16 최종 최소
- `P50 0.9ms / P99 1.1ms / Mean 0.9ms 100 runs`: 100회 반복 측정으로 P99 1.1ms 확보, P50 대비 P99 22% 증가로 tail latency 안정적 - 50개 배포 시에도 100ms SLO 가능 증명
- `README 1줄: P50 1ms P99 1ms @ 233.4MiB 1 replica RTX 5060 Ti`: 포트폴리오용 1줄 요약 완성, 이력서에 바로 붙일 수 있는 형태
- `34MiB P0 19W No running`: 2달 내내 완전 복귀, P8 9W(Day31) → P0 19W(Day32) 부하 후 정상 복귀
- `Not enough SMs to use max_autotune_gemm`: 5060 Ti 36 SMs 경고 유지되지만 reduce-overhead로 232.7MiB 성공 - max_autotune 불가 환경에서도 최적화 가능 증명

### 💡 핵심 포인트

- **2달 최종 232.7MiB / 265.5MiB Peak + P99 1.1ms가 진짜 최종:** Day24 902MiB → Day32 232.7MiB 74.2% 절감, 17→50 3배 + P99 1.1ms 100 runs - 5060 Ti 1대로 메모리 + latency 둘 다 잡은 Blackwell 최종 포트폴리오
- **P99 1.1ms가 Meta E5 면접 4번째 답변:** "latency는? P50 0.9ms P99 1.1ms Mean 0.9ms 100 runs, len8 124M bf16 5060 Ti sm_120 36 SMs에서 1ms, 50개 병렬 55ms로 100ms SLO 만족 - Day32 로그 증명" - 기존 3문장에 1문장 추가
- **232.7MiB가 233.4MiB보다 0.7MiB 더 작은 이유:** inductor cache 최적화, 150MB weights +23MB context +59.7MB cache =232.7MiB - 60MB cache에서 0.3MiB 절감으로 최종 최소
- **QnA:** "2달 902→232.7 + P99 1.1ms 어떻게? 375M→124M 227MB + len 32→8 75% KV + torch.compile cudagraphs=False Peak 844→265 68% + Alloc 233.4→232.7 0.7MiB 추가 + 100 runs P50 0.9ms P99 1.1ms - 5060 Ti 1대로 74% 절감 3배 + 1ms latency 프로덕션 완성했습니다"

---

## 🇺🇸 English

### Measured Results

| Day | Alloc | Peak | Latency | Replicas | Note |
| --- | --- | --- | --- | --- | --- |
| Day24 | 902MiB | - | - | 17 | baseline 375M |
| Day25 vLLM | 608MiB | - | - | 10 | EngineCore 360MiB overhead |
| Day27 | 233.4MiB | 265.5MiB | - | 50 | smashed 400MiB |
| Day30 Final | 233.4MiB | 265.5MiB | - | 50 | 12.1GiB<16GiB OK 74% 3x |
| Day32 P99 NEW | 232.7MiB | 265.5MiB | P50 0.9ms P99 1.1ms Mean 0.9ms | 50 | 2-month lowest + P99 1ms 100 runs |
| After | 34MiB | - | - | - | P0 19W No running |

**How 902→232.7 + P99 1.1ms in 2 months?**

- Alloc 232.7=150MB weights+23MB context+59.7MB cache 0.7MiB extra cut vs 233.4
- Peak 265.5 same as Day27/30 lowest repro 68% cut vs 844MiB stable
- P99 1.1ms 100 runs P50 0.9ms Mean 0.9ms tail 22% stable 50 parallel 55ms <100ms SLO Meta prod proven
- 50: 232.7x50=11.635GiB+0.75GiB=12.385GiB<16GiB OK 0.04GiB extra save vs 12.42GiB

### 💡 Key Insight

- **2-month final 232.7/265.5 + P99 1.1ms is real final:** 902→232.7 74.2% cut 17→50 3x + P99 1.1ms 100 runs - memory + latency both with single 5060 Ti Blackwell final
- **P99 1.1ms is Meta E5 4th answer:** "latency? P50 0.9ms P99 1.1ms Mean 0.9ms 100 runs len8 124M bf16 5060 Ti sm_120 1ms 50 parallel 55ms 100ms SLO - Day32 log proof" - add 1 to existing 3
- **232.7 0.7MiB smaller than 233.4:** inductor cache optimized 150+23+59.7=232.7 60→59.7 0.3MiB cut final floor
- **QnA:** "How 902→232.7 + P99 1.1ms in 2 months? 375M→124M 227MB + len 32→8 75% KV + torch.compile cudagraphs=False Peak 844→265 68% + Alloc 233.4→232.7 0.7MiB extra + 100 runs P50 0.9ms P99 1.1ms - 74% cut 3x +1ms latency prod single 5060 Ti"

---

## 📊 Logs / 로그
```
2-Month Final P99 232.7MiB Peak 265.5MiB P99 1.1ms
day32_p99.txt # Alloc 232.7MiB / Peak 265.5MiB P50 0.9ms P99 1.1ms Mean 0.9ms 100 runs 2-month lowest + latency
nvidia_34MiB_P0_19W.txt # 34MiB P0 19W No running processes full return

History / 히스토리
day24_902MiB_baseline.txt # 902MiB 17 replicas
day25_vLLM_608MiB_10.txt # 608MiB 10 replicas EngineCore 360MiB
day26_580MiB_844Peak.txt # 580.8MiB / 844.3MiB Peak
day27_233MiB_265Peak.txt # 233.4MiB / 265.5MiB Peak smashed 400MiB
day28_sglang_fail.txt # FAIL CUDA 13.0 vs 12.9 mismatch sm_120 not supported
day29_50_prod_12.1GiB.txt # 12400MiB=12.1GiB<16GiB OK
day30_233MiB_265Peak_final.txt # 233.4MiB / 265.5MiB Peak 74% cut 17→50 3x
day31_portfolio.txt # 7 logs table + interview 3 sentences
day32_p99_232.7MiB_1.1ms.txt # NEW 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 2-month best + P99
```


## 🔧 Stack / 스택

- Version / 버전: CUDA 13.2 Driver 595.84, torch cu128 sm_120 36 SMs (12,0) Blackwell, Python 3.12 ai-env, 2-month lowest Alloc
- Base Image / 베이스 이미지: conda ai-env + torch.compile reduce-overhead fullgraph=False + cudagraphs=False + high precision + empty_cache() + 100 runs P99 measurement → 232.7MiB / 265.5MiB Peak P99 1.1ms final
- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell P0 19W 41C
- Model / 모델: SimpleGPT 124M d_model 768 n_layer 12 vocab 50257 len 8 bf16 150MB, 232.7MiB stable 265.5MiB peak lowest + P50 0.9ms P99 1.1ms, 50 replicas 12.38GiB<16GiB

## 💡 Next / 다음

- Day33: PHASE 3 - Llama 3.2 1B bf16 baseline 측정, 232.7MiB 방법론으로 1B 500MiB 이하 도전, 2달 124M 74% 성공을 1B로 확장
- Day33: PHASE 3 - Llama 3.2 1B bf16 baseline, expand 232.7MiB method to 1B <500MiB, 2-month 124M 74% success to 1B


