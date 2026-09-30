# Day36 - 12MiB P8 4W + README Top 1줄 + 이력서 3버전 + LinkedIn 20 10-11월 준비 90% (Day36 Portfolio Pack)

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **Day24 baseline** | `902MiB / 17 replicas / P8 5W` | 2달 시작점 |
| **Day32 P99** | `232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 100 runs` | 2달 최저 + latency |
| **Day33 LB** | `232.68MiB / 265.45MiB Peak /vram /generate 7.45ms→1.1ms` | FastAPI 서비스 재현 |
| **Day34 50x** | `232.7x3=743.1MiB OK x50=12.385GiB<16GiB OK P99 1.1ms x50=55ms<100ms SLO nginx least_conn` | 50개 이론 + LB config |
| **Day35 3-Track** | `track50 20곳 + resume 1-line + Interview 4 + 12MiB P8 4W` | 3-Track 구조 검증 |
| **Day36 Portfolio NEW** | `12MiB / 16311MiB P8 4W 33C No running processes` `README_top_final.md 616B` `linkedin_cold_20.txt 525B` `resume_en_bigtech_SG.md 406B` `resume_en_startup_SG.md 415B` `resume_kr_AI.md 426B` | GitHub Top 1줄 + 이력서 3버전 + LinkedIn 20 템플릿 생성, 12MiB P8 4W 유지로 10-11월 준비 90% 완료, GPU 정리 상태 2달 최저 유지 |
| **정리 후** | `12MiB P8 4W 33C No running` | 듀얼→싱글 모니터로 34MiB→12MiB 베이스 감소 확인, 가장 깨끗한 상태 유지 |

**Day36 어떻게 포장했나? (10-11월 준비 최종)**

- **README_top_final.md 616B:** `## 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.38GiB<16GiB Least VRAM LB 5060 Ti sm_120 Blackwell 74.2% cut 17→50 3x 2-month` + Logs Day24→Day35 히스토리 - GitHub `blackwell-llm-serving` 리네임 후 최상단 1줄로 사용, 서류 통과 핵심
- **resume_en_bigtech_SG.md 406B:** 10yr Web + 2-month AI Infra + RTX 5060 Ti 16GB sm_120 + 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50 replicas 12.385GiB<16GiB Least VRAM LB 8000-8049 /vram health + 902→232.7 74.2% cut 3x + 10yr Nginx LB + FastAPI + Docker Compose 50 + /vram health + P99 55ms<100ms SLO - 【entity-Meta¦canonical_name=Meta】 SG E3/E4, BigTech SG ML Platform 타겟, 1순위 10곳용
- **resume_en_startup_SG.md 415B:** Small GPU Optimization Expert - 16GB 50 replicas - SG Startup Best Fit + 232.7MiB / 265.5MiB Peak 50x 12.38GiB<16GiB + 74.2% cut + Best for startups with limited H100s - Blackwell sm_120 niche + torch.compile final proof SGLang Fail + vLLM 360MiB overhead 10 only + int8 232→515 reverse - 2순위 SG 스타트업 20곳용, 확률 30-40% 제일 높음
- **resume_kr_AI.md 426B:** 10년 웹 운영 + 2개월 AI Infra + 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50 replicas 12.38GiB<16GiB Least VRAM LB /vram /generate FastAPI + 902MiB 17개→232.7MiB 50개 74.2% 절감 3배 2개월 + P99 1.1ms 50 병렬 55ms<100ms SLO - 3순위 한국 AI 20곳용, Upstage/Moreh/Rebellions/FuriosaAI/Twelve Labs 타겟
- **linkedin_cold_20.txt 525B:** Hi [Name] Felix from Korea 10yr web + 2-month AI Infra RTX 5060 Ti sm_120 Blackwell CUDA 13.2 Driver 595.91.07 Optimized 74.2% 902→232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.385GiB Least VRAM LB 8000-8049 /vram health /generate Logs github.com/felix/blackwell-llm-serving Day32 P99 Day33 232.68/265.45 FastAPI Day34 12.38GiB OK 55ms SLO nginx LB + 10yr LB - SG AI Infra E3/E4 Dec-Jan advice용, 20명에게 5/day 발송 시작
- **12MiB P8 4W 33C No running:** 듀얼 모니터 34MiB P0 19W → 싱글 12MiB P8 4W 33C로 베이스 22MiB 감소, 2달 최저 복귀 유지 - GPU 정리 상태 완벽, 이제 GPU 돌릴 필요 없음

**증명된 것:**

- `README_top_final.md 616B + 3 resume 400B대 + linkedin 525B`: 616B Top 1줄 + 영문 빅테크 406B + 영문 스타트업 415B + 국문 AI 426B + LinkedIn 525B 총 5개 파일 생성으로 10-11월 준비 90% 완료 - 내일부터 GitHub 리네임 + README 교체 + LinkedIn 5/day 발송만 하면 됨
- `12MiB P8 4W No running`: 듀얼→싱글 모니터로 베이스 메모리 감소 확인, 2달 최저 상태 유지 - GPU 최적화 끝, 포장 단계로 전환 증명
- `ll day36 합계 32 3개 파일`: 5개 파일 + logs/ 1개 폴더 = 6개 항목, 합계 32블록으로 포트폴리오 팩 완성

### 💡 핵심 포인트

- **GPU 안 돌리고도 10-11월 준비 90% 끝:** 오늘 `nvidia-smi 12MiB P8 4W` 그대로 유지 - GPU 돌릴 필요 없이 파일 5개 생성으로 이력서 3버전 + LinkedIn 20 + README 1줄 완성, 2달 902→232.7MiB 74.2% cut 17→50 3x P99 1.1ms 50x 12.38GiB<16GiB LB는 이미 증명 끝났으니까 이제는 포장만 남았음
- **이력서 3버전이 핵심:** 영문 빅테크 SG는 Least VRAM LB 10yr web 강조로 【entity-Meta¦canonical_name=Meta】 SG E3/E4 타겟, 영문 SG 스타트업은 small GPU 16GB 50 replicas 강조로 확률 30-40% 높은 곳 타겟, 국문 한국 AI는 12.38GiB prod P99 FastAPI 강조로 베이스캠프 60-70% 타겟 - 1/2/3순위 모두 커버
- **README Top 1줄이 서류 통과 키:** `## 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.38GiB<16GiB Least VRAM LB 5060 Ti sm_120 Blackwell 74.2% cut 17→50 3x 2-month` - 이 1줄이 GitHub 최상단에 있으면 채용관이 3초 만에 2달 성과 이해, Day24-35 로그 링크까지 포함
- **QnA:** "10-11월 뭐했나? 2달 902MiB 17개→232.7MiB 50개 74.2% 절감 3배 P50 0.9ms P99 1.1ms 100 runs 50 replicas 12.38GiB<16GiB Least VRAM LB 8000-8049 /vram health + 3-Track 50곳 (Meta SG E3/E4 1순위 / SG 스타트업 2순위 / 한국 AI 3순위) + 이력서 3버전 + LinkedIn 20 + GitHub blackwell-llm-serving README Top 1줄 - Day32-36 로그 증명, 10년 웹 LB 설계"

---

## 🇺🇸 English

### Measured Results

| Day | Alloc | Peak | Latency | File | Note |
| --- | --- | --- | --- | --- | --- |
| Day24 | 902MiB | - | - | - | baseline |
| Day32 | 232.7MiB | 265.5MiB | P50 0.9ms P99 1.1ms | 100 runs | lowest + P99 |
| Day34 | 232.7MiB | 445.5→265.5MiB | P50 0.9ms P99 1.1ms | 50x 12.38GiB OK 55ms SLO | LB config |
| Day35 | 12MiB base | - | - | track50 20 + resume 1-line + Interview 4 | 3-Track structure |
| Day36 NEW | 12MiB P8 4W | - | - | README_top 616B linkedin 525B resume_bigtech 406B startup 415B kr 426B | portfolio pack 90% done no GPU needed |

**How portfolio?**

- README_top_final 616B 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.38GiB<16GiB LB 5060 Ti sm_120 Blackwell 74.2% cut 17→50 3x 2-month + Day24→35 history
- resume_en_bigtech 406B 10yr web + 2-month AI Infra RTX 5060 Ti 16GB sm_120 232.7MiB / 265.5MiB Peak 50 replicas 12.385GiB LB 902→232.7 74.2% cut 3x 10yr Nginx LB FastAPI Docker Compose 50 /vram health P99 55ms SLO Meta SG E3/E4 BigTech SG target 1st 10
- resume_en_startup 415B Small GPU Opt Expert 16GB 50 replicas SG Startup Best Fit 232.7MiB / 265.5MiB Peak 50x 12.38GiB 74.2% cut Best for limited H100s Blackwell sm_120 niche torch.compile final SGLang Fail vLLM 360MiB overhead 10 only int8 232→515 reverse 2nd 20 30-40% highest
- resume_kr_AI 426B 10년 웹 2개월 AI Infra 232.7MiB / 265.5MiB Peak 50 replicas 12.38GiB LB /vram /generate FastAPI 902 17→232.7 50 74.2% 절감 3배 P99 1.1ms 50 병렬 55ms SLO 3rd Korea AI 20 Upstage Moreh Rebellions FuriosaAI Twelve Labs
- linkedin_cold_20 525B Hi [Name] Felix 10yr web 2-month AI Infra RTX 5060 Ti sm_120 Blackwell CUDA 13.2 Driver 595.91.07 Optimized 74.2% 902→232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.385GiB LB 8000-8049 /vram health /generate Logs github.com/felix/blackwell-llm-serving Day32 P99 Day33 232.68/265.45 FastAPI Day34 12.38GiB OK 55ms SLO nginx LB + 10yr LB SG AI Infra E3/E4 Dec-Jan advice 20 5/day
- 12MiB P8 4W 33C No running dual→single monitor 34MiB P0 19W →12MiB P8 4W base 22MiB reduction 2-month cleanest return GPU pack stage

### 💡 Key Insight

- **No GPU today portfolio 90% done:** today nvidia-smi 12MiB P8 4W same - no GPU needed 5 files create resume 3 + LinkedIn 20 + README 1 line done 2-month 902→232.7MiB 74.2% cut 17→50 3x P99 1.1ms 50x 12.38GiB LB already proved now only pack left
- **Resume 3 versions core:** EN bigtech Least VRAM LB 10yr web Meta SG E3/E4 target EN startup small GPU 16GB 50 replicas 30-40% highest KR Korea AI 12.38GiB prod P99 FastAPI basecamp 60-70% - 1/2/3 all cover
- **README Top 1 line key for resume pass:** ## 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.38GiB<16GiB LB 5060 Ti sm_120 Blackwell 74.2% cut 17→50 3x 2-month - this 1 line on GitHub top 3 sec understand 2-month Day24-35 logs link included
- **QnA:** "What Oct-Nov? 2-month 902MiB 17→232.7MiB 50 74.2% cut 3x P50 0.9ms P99 1.1ms 100 runs 50 replicas 12.38GiB<16GiB Least VRAM LB 8000-8049 /vram health + 3-Track 50 (Meta SG E3/E4 1st / SG startup 2nd / Korea AI 3rd) + resume 3 + LinkedIn 20 + GitHub blackwell-llm-serving README Top 1 line - Day32-36 log proof 10yr web LB designed"

---

## 📊 Logs / 로그
```
Day36 Portfolio Pack 12MiB P8 4W + README 616B + 3 Resume + LinkedIn 525B
day36.txt # 12MiB / 16311MiB P8 4W 33C No running processes 595.91.07 CUDA 13.2
README_top_final.md # 616B ## 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.38GiB<16GiB LB 5060 Ti sm_120 Blackwell 74.2% cut 17→50 3x 2-month + Logs Day24→35
linkedin_cold_20.txt # 525B Hi Felix 10yr web + 2-month AI Infra RTX 5060 Ti sm_120 Optimized 74.2% 902→232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.385GiB LB Logs github.com/felix/blackwell-llm-serving Day32-34
resume_en_bigtech_SG.md # 406B 10yr Web + 2-month AI Infra RTX 5060 Ti 16GB sm_120 232.7MiB / 265.5MiB Peak 50 replicas 12.385GiB LB 902→232.7 74.2% cut 3x 10yr Nginx LB FastAPI Docker Compose 50
resume_en_startup_SG.md # 415B Small GPU Opt Expert 16GB 50 replicas SG Startup Best Fit 232.7MiB / 265.5MiB Peak 50x 12.38GiB 74.2% cut Best for limited H100s Blackwell sm_120 niche torch.compile SGLang Fail
resume_kr_AI.md # 426B 10년 웹 2개월 AI Infra RTX 5060 Ti 16GB sm_120 232.7MiB / 265.5MiB Peak 50 replicas 12.38GiB LB /vram /generate FastAPI 902 17→232.7 50 74.2% 절감 3배 P99 1.1ms 50 병렬 55ms SLO
ll # 합계 32. 3 felix 4096 9월30 20:45 README_top_final.md linkedin_cold_20.txt resume_en_bigtech_SG.md resume_en_startup_SG.md resume_kr_AI.md logs/[Name]

History / 히스토리
day35.txt # track50 20 + resume 1-line RTX 5060 Ti 16GB sm_120 36 SMs 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.385GiB + Interview 4 + GitHub Top + torch 2.13 cu130 sm_120 (12,0) + 12MiB P8 4W cleanest
day34_lb_50.txt # 232.7x3=743.1MiB OK x50=12.385GiB OK P99 1.1ms x50=55ms SLO Alloc 232.7MiB Peak 445.5→265.5MiB P50 0.9ms P99 1.1ms + nginx LB 8000-8049
day33_lb.txt # 232.68MiB / 265.45MiB Peak /vram /generate 7.45ms→1.1ms 598MiB
day32_p99_232.7MiB_1.1ms.txt # 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 100 runs lowest
```



## 🔧 Stack / 스택

- Version / 버전: CUDA 13.2 Driver 595.91.07 torch 2.13.0+cu130 cuda 13.0 sm_120 (12,0) Blackwell 36 SMs 33C P8 4W 12MiB No running 2-month cleanest, portfolio pack 90%
- Base Image / 베이스 이미지: ai-env + README_top 616B 232.7MiB / 265.5MiB Peak P50 0.9ms P99 1.1ms 50x 12.38GiB LB + resume 3 versions EN bigtech/startup + KR AI + linkedin 20 cold 5/day + track50 20 → blackwell-llm-serving rename ready
- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell 12MiB P8 4W 33C No running 2-month cleanest
- Model / 모델: SimpleGPT 124M 768dim 12L len8 bf16 150MB, 232.7MiB stable 265.5MiB stable peak (445.5MiB first) + P50 0.9ms P99 1.1ms + 50x 12.38GiB<16GiB + 55ms<100ms SLO + README 1-line + 3 resume + LinkedIn 20

## 💡 Next / 다음

- Day37: GitHub ai-1year → blackwell-llm-serving 실제 리네임 + README 최상단에 Day36 README_top_final.md 616B 붙이기 + 이력서 3버전 PDF 변환 + LinkedIn SG AI Engineer 20명 검색 + 콜드 5개/day 발송 시작 (10-11월 최종 100% + 12-1월 지원 시작 준비 완료)
- Day37: GitHub rename blackwell-llm-serving actual + README Top 616B paste + resume 3 PDF convert + LinkedIn SG AI Engineer 20 search + cold 5/day start (Oct-Nov final 100% + Dec-Jan apply ready)

