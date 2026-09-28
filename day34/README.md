# Day34 - 232.7MiB Peak 445.5MiB→265.5MiB P50 0.9ms P99 1.1ms 50x 12.38GiB LB Config (Day34 LB Final)

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **Day24 baseline** | `902MiB / 17 replicas / P8 5W / 375M 757MB` | 2달 시작점 |
| **Day25 vLLM** | `608MiB / 10 replicas / EngineCore 360MiB overhead` | 10개만 |
| **Day27 233MiB** | `233.4MiB / 265.5MiB Peak / smashed 400MiB` | 74% 절감 |
| **Day29 50 prod** | `233.4MiB / 445.5MiB Peak / 12400MiB=12.1GiB<16GiB OK` | 첫 컴파일 Peak 445MiB 1회성 |
| **Day30 Final** | `233.4MiB / 265.5MiB Peak / 12422MiB=12.1GiB<16GiB OK / 74% cut 17→50 3x` | 2달 베스트 |
| **Day32 P99** | `232.7MiB / 265.5MiB Peak / P50 0.9ms P99 1.1ms Mean 0.9ms 100 runs` | 2달 최저 Alloc 0.7MiB 절감 + P99 1.1ms |
| **Day33 LB** | `232.68MiB / 265.45MiB Peak / /vram 232.68 / 265.45 / /generate 7.45ms first →1.1ms / 598MiB` | FastAPI 서비스 100% 재현 |
| **Day34 50x NEW** | `Not enough SMs to use max_autotune_gemm / 232.7MiB x3=698.1+45=743.1MiB<16311MiB OK / 232.7x50=11.635+0.75=12.385GiB<16GiB OK / P99 1.1ms x50=55ms<100ms SLO / Alloc 232.7MiB Peak 445.5MiB P50 0.9ms P99 1.1ms 50x Theory 12.38GiB OK` `nginx upstream least_conn 8000-8049 /vram /` | 50개 이론 검증 완성, Peak 445.5MiB 첫 컴파일 1회성 후 265.5MiB 안정화, 3개 743MiB <16GiB + 50개 12.38GiB <16GiB OK + 50 parallel 55ms <100ms SLO + LB config 완성 |
| **정리 후** | `34MiB / 16311MiB P0 19W 40C No running processes` | 2달 내내 완전 복귀, Driver 595.91.07 CUDA 13.2 |

**Day34 50개 어떻게 검증했나? (10-11월 취업 준비 Track 2 최종)**

- **Not enough SMs to use max_autotune_gemm:** 5060 Ti 36 SMs (12,0) Blackwell sm_120에서 max_autotune_gemm 불가 경고 유지, reduce-overhead로도 232.7MiB 성공 - max_autotune 불가 환경 최적화 가능 증명
- **232.7MiB x3 = 698.1MiB + 15MiB x3 = 743.1MiB <16311MiB OK:** 3개 replica 동시 구동 743MiB <16GiB 검증, 실제로는 1개로 100 runs 측정했지만 이론 계산으로 3개도 OK 증명
- **232.7MiB x50 = 11.635GiB + 0.75GiB = 12.385GiB <16GiB OK (Day32 갱신):** Day30 12.42GiB 대비 Day32 232.7MiB 갱신으로 12.38GiB 0.04GiB 추가 절감, 50개 동시 A/B 테스트 프로덕션 가능 최종 증명
- **P99 1.1ms x50 parallel = 55ms <100ms SLO:** P50 0.9ms P99 1.1ms 100 runs가 50개 병렬로도 55ms로 100ms SLO 만족 - 【entity-Meta¦canonical_name=Meta】 프로덕션 latency 요구 충족, Tail latency 안정적
- **Alloc 232.7MiB Peak 445.5MiB P50 0.9ms P99 1.1ms 50x Theory 12.38GiB OK:** Alloc 232.7MiB 2달 최저 유지, Peak 445.5MiB는 첫 inductor 컴파일 1회성 (Day29 동일), empty_cache() 후 265.5MiB로 안정화 - Day27/32/33 265.5MiB 동일 재현
- **nginx upstream least_conn 8000-8049 /vram /:** Least VRAM LB config 완성, 8000-8049 50개 포트 + /vram health check + / proxy_pass - 10년 웹 경력 Nginx LB가 AI Infra에서 그대로 통하는 차별점

**증명된 것:**

- `232.7MiB x50 = 12.385GiB <16GiB OK`: 2달 최적화 232.7MiB로 50개 12.38GiB <16GiB 최종 증명, 17→50 3배 증가 74.2% 절감으로 50개 A/B 테스트 프로덕션 완성
- `P99 1.1ms x50 parallel = 55ms <100ms SLO`: P99 1.1ms 100 runs가 50개 병렬 55ms로 100ms SLO 만족 - latency 포트폴리오 완성, 이력서 1줄에 바로 쓸 수 있는 숫자
- `Peak 445.5MiB → 265.5MiB 안정화`: 첫 컴파일 445.5MiB 1회성 후 empty_cache()로 265.5MiB 안정화, Day29 445.5MiB와 동일한 패턴 - 첫 호출 7.45ms와 같은 warmup 특성, 면접에서 설명 가능
- `upstream least_conn 8000-8049 /vram /`: Nginx LB config 완성으로 50개 프로덕션 구조 설계 완료 - 1순위 Meta SG E3/E4 / 2순위 SG 스타트업 / 3순위 한국 AI 모두에 먹히는 구조
- `34MiB P0 19W No running processes`: 2달 내내 완전 복귀, Driver 595.84→595.91.07 업데이트 후에도 동일

### 💡 핵심 포인트

- **50개 12.38GiB<16GiB OK + P99 55ms<100ms SLO가 최종:** 232.7MiB x50=11.635GiB+0.75GiB=12.385GiB<16GiB + P99 1.1ms x50=55ms<100ms SLO - 메모리 + latency 둘 다 50개 프로덕션 가능, 2달 902MiB 17개 → 232.7MiB 50개 3배 + 74.2% 절감 + P99 1.1ms 최종 포트폴리오 완성
- **Peak 445.5MiB는 첫 컴파일 1회성, 안정 Peak는 265.5MiB:** Day29 445.5MiB와 Day34 445.5MiB 동일한 첫 컴파일 Peak, empty_cache() 후 265.5MiB로 안정화 - Day27/32/33 265.5MiB 3번 재현으로 안정 Peak 증명, 면접에서 "왜 Peak가 두 개냐? 첫 inductor compile 445MiB 1회성 후 265MiB 안정화 Day29/34 로그 증명" 답변 완성
- **Nginx LB config가 10년 웹 경력 차별점:** `least_conn 8000-8049 /vram health / proxy_pass` - AI Infra 면접관이 "50개 어떻게 배포?" 물어보면 이 config 바로 보여주면 10년 경력 증명, SG 스타트업이 제일 좋아하는 포인트
- **QnA:** "50개 latency는? Alloc 232.7MiB Peak 445.5MiB 첫 컴파일 후 265.5MiB 안정화 P50 0.9ms P99 1.1ms 100 runs, 232.7x50=12.38GiB<16GiB OK + P99 1.1ms x50=55ms<100ms SLO + Nginx least_conn 8000-8049 /vram health - Day32-34 로그 증명, 10년 웹 LB 경력으로 설계했습니다"

---

## 🇺🇸 English

### Measured Results

| Day | Alloc | Peak | Latency | Theory | Note |
| --- | --- | --- | --- | --- | --- |
| Day24 | 902MiB | - | - | 17 | baseline |
| Day27 | 233.4MiB | 265.5MiB | - | 50 | 74% cut |
| Day30 | 233.4MiB | 265.5MiB | - | 12.42GiB<16GiB 3x | final best |
| Day32 | 232.7MiB | 265.5MiB | P50 0.9ms P99 1.1ms | 12.38GiB<16GiB | lowest + P99 |
| Day33 | 232.68MiB | 265.45MiB | 7.45ms first→1.1ms | /vram /generate | service repro |
| Day34 NEW | 232.7MiB | 445.5MiB→265.5MiB | P50 0.9ms P99 1.1ms | 698MiB 3x OK 12.385GiB 50x OK 55ms<100ms SLO | LB config + 50x theory final |
| After | 34MiB | - | - | - | P0 19W No running |

**How 50x?**

- Not enough SMs max_autotune_gemm warning 36 SMs Blackwell sm_120 reduce-overhead still 232.7MiB success
- 232.7x3=698.1+45=743.1MiB<16311MiB OK 3 replicas
- 232.7x50=11.635+0.75=12.385GiB<16GiB OK Day32 renew 0.04GiB extra save vs 12.42GiB
- P99 1.1ms x50=55ms<100ms SLO latency portfolio final
- Alloc 232.7 Peak 445.5 first compile 1 time then 265.5 stable Day29 same pattern empty_cache()
- nginx upstream least_conn 8000-8049 /vram / - 10yr web LB diff point

### 💡 Key Insight

- **50x 12.38GiB<16GiB OK + P99 55ms<100ms SLO final:** 232.7x50=12.38GiB<16GiB + P99 1.1ms x50=55ms<100ms SLO - memory + latency both 50 prod possible 902 17→232.7 50 3x 74.2% + P99 1.1ms final portfolio
- **Peak 445.5 first compile 1 time stable 265.5:** Day29 445.5 same Day34 445.5 first inductor compile then 265.5 stable Day27/32/33 265.5 3 repro stable peak proof "why 2 peaks? first 445MiB 1 time then 265MiB stable Day29/34 log"
- **Nginx LB config 10yr web diff:** least_conn 8000-8049 /vram health / proxy_pass - AI Infra interview "how 50 deploy?" show this config 10yr proof SG startup favorite
- **QnA:** "50 latency? Alloc 232.7 Peak 445.5 first then 265.5 stable P50 0.9ms P99 1.1ms 100 runs 232.7x50=12.38GiB<16GiB OK + P99 1.1ms x50=55ms<100ms SLO + Nginx least_conn 8000-8049 /vram health - Day32-34 log proof 10yr web LB designed"

---

## 📊 Logs / 로그
```
Day34 LB Final 232.7MiB Peak 445.5→265.5 P99 1.1ms 50x 12.38GiB OK
day34_lb_50.txt # Not enough SMs max_autotune_gemm / 232.7MiB x3=698.1+45=743.1MiB<16311MiB OK / 232.7x50=11.635+0.75=12.385GiB<16GiB OK / P99 1.1ms x50=55ms<100ms SLO / Alloc 232.7MiB Peak 445.5MiB P50 0.9ms P99 1.1ms 50x Theory 12.38GiB OK
nvidia_34MiB_P0_19W.txt # 34MiB / 16311MiB P0 19W 40C No running processes 595.91.07 CUDA 13.2
nginx-lb.conf # upstream llm_backend least_conn 8000-8049 /vram / proxy_pass 50 replicas LB config

History / 히스토리
day33_lb.txt # 232.68MiB / 265.45MiB Peak /vram /generate 7.45ms first→1.1ms 598MiB
day32_p99_232.7MiB_1.1ms.txt # 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 100 runs 2-month lowest
day31_portfolio.txt # 7 logs table + interview 3 sentences
day30_233MiB_265Peak_final.txt # 233.4MiB / 265.5MiB Peak 74% cut 17→50 3x
```



## 🔧 Stack / 스택

- Version / 버전: CUDA 13.2 Driver 595.91.07 (595.84→595.91.07 업뎃) torch cu128 sm_120 36 SMs (12,0) Blackwell, max_autotune_gemm 불가 reduce-overhead로 232.7MiB 성공, 50x Theory 12.38GiB OK
- Base Image / 베이스 이미지: conda ai-env + torch.compile reduce-overhead fullgraph=False + cudagraphs=False + high precision + empty_cache() + 100 runs P99 + docker-compose 50 + nginx least_conn LB → 232.7MiB / 265.5MiB stable Peak P99 1.1ms 50x 12.38GiB OK
- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell P0 19W 40C 34MiB after
- Model / 모델: SimpleGPT 124M 768dim 12L len8 bf16 150MB, 232.7MiB stable 265.5MiB stable peak (445.5MiB first compile 1 time) + P50 0.9ms P99 1.1ms + 50x 12.38GiB<16GiB + P99 50 parallel 55ms<100ms SLO + LB 8000-8049

## 💡 Next / 다음

- Day35: 3-Track 50곳 리스트 (1순위 빅테크 SG E3/E4 10곳 + 2순위 SG 스타트업 20곳 + 3순위 한국 AI 20곳) + 이력서 3버전 (영문 빅테크/SG 스타트업 + 국문 한국) + LinkedIn 콜드 메시지 20개 템플릿, 10-11월 준비 최종 + 12-1월 지원 시작
- Day35: 3-Track 50 companies (1st BigTech SG E3/E4 10 + 2nd SG startup 20 + 3rd Korea AI 20) + resume 3 versions + LinkedIn 20 cold messages, Oct-Nov prep final + Dec-Jan apply start

