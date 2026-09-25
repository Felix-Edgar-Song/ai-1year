# Day31 - 2달 최종 포트폴리오 패키징 902MiB→233MiB 74% 50개 12.1GiB (2-Month Final Portfolio)

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **Day24 baseline** | `902MiB / 17 replicas / P8 5W / 375M 757MB` | python:3.12-slim + torch cu128 sm_120 네이티브, 2달 시작점 |
| **Day25 vLLM** | `608MiB / 10 replicas / EngineCore 360MiB overhead` | PagedAttention 90.5% 절감 시도, EngineCore 고정 오버헤드로 10개만 |
| **Day26 torch.compile 580** | `580.8MiB / 844.3MiB Peak / 24L 1024dim` | 첫 torch.compile, 24L 1024dim inductor cache 180MiB |
| **Day27 233MiB** | `233.4MiB / 265.5MiB Peak / 12L 768dim / smashed 400MiB` | 12L 768dim len8 cudagraphs=False로 Peak 844→265MiB 68% 절감, 400MiB 목표 167MiB 초과 달성 |
| **Day28 SGLang** | `FAIL CUDA 13.0 vs 12.9 mismatch / sm_120 not supported / torch.compile final` | Blackwell sm_120 flashinfer 미지원 증명, SGLang import 불가로 torch.compile 233MiB 최종 정답 |
| **Day29 50 prod** | `233.4MiB / 445.5MiB Peak / 12400MiB=12.1GiB<16GiB OK / Least VRAM LB` | 50개 프로덕션 이론 검증, Least VRAM LB 8000-8049, 첫 컴파일 445MiB 1회성 |
| **Day30 Final** | `233.4MiB / 265.5MiB Peak / 12422MiB=12.1GiB<16GiB OK / 74% cut 17→50 3x / 2-month best` | Day24 902→233 74% 절감 17→50 3배 2달 최종 베스트, Peak 265MiB 최저 재현 |
| **Day31 Portfolio** | `비교표 7줄 + 면접 3문장 + 34MiB P8 9W No running` | 2달 로그 7개 모아 ai-1year/README.md 최상단 패키징, Meta 포트폴리오 최종 완성 |
| **정리 후** | `34MiB / 16311MiB P8 9W 39C No running processes` | Xorg 34MiB 포함 완전 복귀, 2달 내내 동일 정리 상태 |

**2달 902→233MiB 어떻게 했나? (전체 타임라인)**

- Day0-24: Numpy 1-5 → 375M 757MB 902MiB baseline, Docker python:3.12-slim + torch cu128 sm_120 네이티브, 17개 이론
- Day25: vLLM 608MiB / 10 replicas - PagedAttention EngineCore 360MiB 고정 오버헤드로 902MiB 17개보다 적은 10개로 실패
- Day26: torch.compile 580.8MiB / 844.3MiB Peak 24L 1024dim - reduce-overhead 첫 적용, inductor cache 180MiB 확인
- Day27: 375M→124M 축소(d_model 1024→768, n_layer 24→12, len 32→8) 227MB 절감 + cudagraphs=False + empty_cache() + high precision으로 233.4MiB / 265.5MiB Peak, Peak 68% 절감
- Day28: SGLang FAIL `CUDA 13.0 vs 12.9 mismatch` - Driver 595.84 CUDA 13.2 환경에서 PyTorch cu128 13.0 vs torchvision cu129 12.9 충돌, flashinfer sm_120 미지원 증명, torch.compile 최종 정답
- Day29: 50개 프로덕션 `12400MiB=12.1GiB<16GiB OK` - 233MiB x50 + FastAPI 15MiB overhead = 12.1GiB 검증, Least VRAM LB 8000-8049 포트
- Day30-31: `233.4MiB / 265.5MiB Peak` 최저 재현 + `12422MiB=12.1GiB<16GiB OK` + `74% cut 17→50 3x` 2달 최종 베스트 확정 + 비교표 7줄 + 면접 3문장 패키징

**증명된 것:**

- `Day24 902MiB 17 vs Day25 608MiB 10 vs Day27-30 233.4MiB 50`: 17→10→50 3배 증가, 902→233 74% 절감 - 5060 Ti 1대로 2달 최적화 최종
- `Day28 FAIL sm_120 not supported / torch.compile final`: Blackwell 36 SMs (12,0) sm_120에서 SGLang flashinfer 미지원 + vLLM 360MiB 고정 10개만 - torch.compile 233MiB Peak 265MiB 유일 <400MiB +50개 해법 2달 실험으로 증명
- `Day30 Final 233.4MiB / 265.5MiB Peak / 12422MiB=12.1GiB<16GiB OK / 74% cut 17→50 3x / 2-month best`: 2달 성과 한 줄 요약, 150MB 가중치 +23MB context +60MB cache =233MiB 이론 100% 일치
- `34MiB P8 9W No running processes`: 2달 내내 동일한 완전 복귀 상태, Xorg 포함 정상
- `Meta E5 3문장`: 1. 50개 12.1GiB<16GiB Least VRAM LB 8000-8049 /vram health 2. int8 232→515MiB 증가 7B+만 효과 Day27 로그 3. torch.compile sm_120 미지원 + vLLM 360MiB 고정 10개만 233MiB Peak 265MiB 유일 - 30초 답변 완성

### 💡 핵심 포인트

- **2달 최종 902MiB 17개 → 233.4MiB Peak 265.5MiB 50개 12.1GiB 3배:** Day24 baseline 902MiB P8 5W 375M → Day31 233MiB P8 9W 124M, 74% 절감 + 50개 동시 A/B 테스트 프로덕션 가능 - 5060 Ti 16GB 16311MiB 1대로 해낸 Blackwell sm_120 최적화 최종 포트폴리오
- **SGLang Fail + int8 역전이 2달 킬러 증명:** Day28 `CUDA 13.0 vs 12.9 mismatch` flashinfer sm_120 미지원 + Day27 `232→515MiB 증가 quantized::linear_dynamic CPU only`로 bf16 233MiB가 Blackwell 최소 증명, 7B+만 int8 효과 - Meta E5 단골 질문 2개 해결
- **50개 12.1GiB<16GiB OK가 프로덕션 능력:** 233MiB x50=11.6GiB + FastAPI 15MiB x50=0.75GiB=12.1GiB <16GiB, Least VRAM LB + /vram health check 8000-8049 포트 - 10년 웹 운영 경력으로 로드밸런싱 설계는 네가 제일 잘하는 분야
- **QnA:** "2달 902→233 어떻게? 375M→124M 227MB 절감 + len 32→8 75% KV 절감 + torch.compile cudagraphs=False Peak 844→265 68% 절감 + SGLang Blackwell Fail torch.compile 최종 + 50개 12.1GiB<16GiB Least VRAM LB - 5060 Ti 1대로 74% 절감 3배 증가 프로덕션 완성했습니다"

---

## 🇺🇸 English

### Measured Results

| Day | Alloc | Peak | Replicas | Note |
| --- | --- | --- | --- | --- |
| Day24 | 902MiB | - | 17 | baseline 375M |
| Day25 vLLM | 608MiB | - | 10 | EngineCore 360MiB overhead |
| Day26 | 580.8MiB | 844.3MiB | - | 24L 1024dim first torch.compile |
| Day27 | 233.4MiB | 265.5MiB | 50 | 74% cut smashed 400MiB |
| Day28 | FAIL | - | - | CUDA 13.0 vs 12.9 mismatch sm_120 not supported |
| Day29 | 233.4MiB | 445.5MiB | 50 | 12400MiB=12.1GiB<16GiB OK Least VRAM LB |
| Day30 Final | 233.4MiB | 265.5MiB | 50 | 12422MiB=12.1GiB<16GiB OK 74% cut 17→50 3x 2-month best |
| Day31 Portfolio | - | - | - | 7 logs table + interview 3 sentences + 34MiB P8 9W |
| After cleanup | 34MiB | - | - | P8 9W No running processes |

**How 902→233 74% in 2 months?**

- Day0-24: Numpy 1-5 →375M 757MB 902MiB baseline 17 replicas python:3.12-slim torch cu128 sm_120
- Day25: vLLM 608MiB 10 EngineCore 360MiB overhead fail less than 17
- Day26: 580.8/844.3 Peak 24L 1024dim first torch.compile inductor 180MiB
- Day27: 375M→124M shrink 227MB + len 32→8 75% KV + cudagraphs=False Peak 844→265 68% cut smashed 400MiB
- Day28: SGLang FAIL mismatch sm_120 not supported torch.compile final
- Day29: 50 prod 12400MiB=12.1GiB<16GiB OK Least VRAM LB 8000-8049
- Day30-31: 233.4/265.5 Peak lowest repro 12422MiB=12.1GiB<16GiB OK 74% cut 17→50 3x 2-month best + 7 logs table packaging

### 💡 Key Insight

- **2-month final 902MiB 17 →233.4MiB Peak 265.5MiB 50 12.1GiB 3x:** 74% cut + 50 A/B prod possible with single 5060 Ti 16GB 16311MiB - Blackwell sm_120 optimization final portfolio
- **SGLang Fail + int8 reverse killer proof:** Day28 CUDA mismatch flashinfer sm_120 not supported + Day27 232→515MiB increase CPU only bf16 233MiB floor 7B+ only int8 - Meta E5 2 questions solved
- **50 12.1GiB<16GiB OK is prod skill:** 233x50=11.6GiB +15MiB x50=0.75GiB=12.1GiB <16GiB Least VRAM LB /vram health 8000-8049 ports - 10yr web ops LB best skill
- **QnA:** "How 902→233 in 2 months? 375M→124M 227MB + len 32→8 75% KV + torch.compile cudagraphs=False Peak 844→265 68% + SGLang Fail final +50 12.1GiB<16GiB Least VRAM LB - 74% cut 3x prod single 5060 Ti"

---

## 📊 Logs / 로그
```
2-Month Final Portfolio 233MiB Peak 265MiB 74% 50 replicas 12.1GiB
day31_portfolio.txt # Day24 902MiB 17 vs Day25 608MiB 10 vs Day27-30 233.4MiB 50 =74% cut 3x, 7 logs table + interview 3 sentences
nvidia_34MiB_P8_9W_return.txt # 34MiB P8 9W 39C No running processes full return 2-month consistent

History / 히스토리
day24_902MiB_baseline.txt # 902MiB P8 5W 375M 757MB 17 replicas start
day25_vLLM_608MiB_10.txt # 608MiB 10 replicas EngineCore 360MiB overhead
day26_580MiB_844Peak.txt # 580.8MiB / 844.3MiB Peak 24L 1024dim
day27_233MiB_265Peak.txt # 233.4MiB / 265.5MiB Peak cudagraphs=False smashed 400MiB
day28_sglang_fail.txt # FAIL CUDA 13.0 vs 12.9 mismatch sm_120 not supported torch.compile final
day29_50_prod_12.1GiB.txt # 233.4MiB / 445.5MiB Peak / 12400MiB=12.1GiB<16GiB OK Least VRAM LB
day30_233MiB_265Peak_final.txt # 233.4MiB / 265.5MiB Peak 12422MiB=12.1GiB<16GiB OK 74% cut 17→50 3x 2-month best
day31_portfolio_final.txt # 7 logs table + Meta E5 3 sentences portfolio packaging
```



## 🔧 Stack / 스택

- Version / 버전: CUDA 13.2 Driver 595.84, torch cu128 sm_120 36 SMs (12,0) Blackwell, Python 3.12 ai-env meta-ai conda, SGLang flashinfer sm_120 not supported CUDA 13.0 vs 12.9 mismatch, vLLM EngineCore 360MiB overhead
- Base Image / 베이스 이미지: conda ai-env + torch.compile mode reduce-overhead fullgraph=False + torch._inductor.config.triton.cudagraphs=False + torch.set_float32_matmul_precision('high') + empty_cache() → 233.4MiB / 265.5MiB Peak final
- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell P0/P8 9W 39C
- Model / 모델: SimpleGPT 124M d_model 768 n_layer 12 vocab 50257 len 8 bf16 150MB, 233MiB stable 265MiB peak lowest, 50 replicas 12.1GiB <16GiB, 17→50 3x

## 💡 Next / 다음

Day32: PHASE 3 Kickoff - Llama 3.2 1B를 233MiB 방식으로 500MiB 이하 도전, 2달 124M 성공을 1B로 확장
Day32: PHASE 3 Kickoff - Llama 3.2 1B with 233MiB method <500MiB challenge, expand 2-month 124M success to 1B

