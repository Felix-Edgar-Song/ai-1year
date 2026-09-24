# Day30 - 2달 최종 233MiB Peak 265MiB 74% 절감 50개 프로덕션 (2-Month Finale)

## 🇰🇷 한국어

### 실측 결과

| 항목 | 실측 값 | 의미 |
| --- | --- | --- |
| **Day0-24 baseline** | `902MiB P8 5W` `375M 757MB 17개` `Numpy 1-5` | python:3.12-slim + torch cu128 sm_120 네이티브, 2달 시작 |
| **Day25 vLLM** | `608MiB` `124M 10개` `90.5% 절감 실패` | PagedAttention EngineCore 360MiB 고정 오버헤드, 10개만 가능 |
| **Day26 580MiB** | `580.8MiB / 844.3MiB Peak` `24L 1024dim` | 첫 torch.compile, inductor cache 180MiB |
| **Day27 233MiB** | `233.4MiB / 265.5MiB Peak` `cudagraphs=False` | 12L 768dim len8, Peak 844→265MiB 68% 절감, 400MiB 목표 167MiB 초과 |
| **Day28 SGLang** | `CUDA True SM (12,0) / SGLang fail CUDA 13.0 vs 12.9 mismatch` | Blackwell sm_120 flashinfer 미지원 증명, torch.compile 최종 정답 |
| **Day29 50개** | `233.4MiB / 445.5MiB Peak / 12400MiB=12.1GiB OK` | 50개 프로덕션 이론 검증, Least VRAM LB 8000-8049 |
| **Day30 Final 2달** | `233.4MiB Alloc / 265.5MiB Peak P0 20W` `12422MiB=12.1GiB <16GiB OK` `74% 절감 17→50 3배` | Day24 902MiB → Day30 233MiB 74% 절감 최종 확정, Peak 265MiB 최저 재현, 50개 A/B 프로덕션 가능 |
| **int8 역전** | `232MiB → 515.7MiB 증가` `quantized::linear_dynamic CPU only` | 124M 작은 모델 int8 비효율 증명, 7B+만 효과 - 면접 킬러 답변 |
| **정리 후** | `34MiB P0 20W No running processes` `41C` | Xorg 34MiB 포함 완전 복귀, 2달 내내 동일한 정리 상태 유지 |

**2달 902MiB→233MiB 74% 절감 어떻게 했나? (PHASE 1 → PHASE 2)**

- PHASE 1 (Day0-24): Numpy 1-5로 시작, Day24 375M 757MB 902MiB baseline 측정 - python:3.12-slim + torch cu128 sm_120 네이티브 확인
- PHASE 2 Day25: vLLM 608MiB 10개 - PagedAttention 시도, EngineCore 360MiB 고정 오버헤드로 실패, 902MiB 17개보다 적은 10개로 역전
- PHASE 2 Day26-27: 375M→124M 축소(d_model 1024→768, n_layer 24→12, len 32→8) 227MB 절감 + torch.compile reduce-overhead + fullgraph=False + cudagraphs=False + empty_cache() + torch.set_float32_matmul_precision('high')로 580/844 Peak → 233/265 Peak 68% 절감
- Day28-29: SGLang fail `CUDA 13.0 vs 12.9 mismatch`로 Blackwell sm_120 flashinfer 미지원 증명 + int8 232→515MiB 증가로 bf16 최소 증명 + 50개 12.1GiB <16GiB OK Least VRAM LB 검증
- Day30 Final: 233.4MiB / 265.5MiB Peak 재현으로 2달 최고 기록 확정, 902→233 74% 절감 + 17→50 3배 증가 = Meta 프로덕션 포트폴리오 완성

**증명된 것:**

- `Final Day30: 233.4MiB / 265.5MiB Peak`: Day27 최저 Peak 재현, Day29 445MiB 대비 180MiB 추가 절감, 2달 최종 최소치 확정
- `Day24 902MiB → Day30 233.4MiB = 74% 절감, 17개 → 50개 3배`: 2달 성과 한 줄 요약, PHASE 1 902MiB 17개에서 PHASE 2 233MiB 50개로 메모리 74% 절감 + 동시 구동 3배 증가
- `50개: 12422MiB = 12.1GiB <16GiB OK`: FastAPI 15MiB overhead 포함 실측, 16311MiB GPU에서 50개 A/B 테스트 프로덕션 가능 - Meta E5 "50개 어떻게 배포?" 답변 증명
- `34MiB P0 20W No running processes`: 2달 내내 동일한 정리 상태, Day24 12MiB(터미널만) → Day30 34MiB(Xorg+GNOME) 정상, 완전 복귀

### 💡 핵심 포인트

- **2달 최종 233.4MiB / 265.5MiB Peak 74% 절감 50개 3배:** Day0 Numpy 1 → Day30 233MiB, 902MiB 17개 → 50개 12.1GiB <16GiB로 3배 증가, Peak 844→265MiB 68% 절감으로 프로덕션 안정성 확보 - 5060 Ti 1대로 해낸 Blackwell 최적화 최종
- **SGLang Fail + vLLM 608MiB 10개 vs torch.compile 233MiB 50개:** Blackwell sm_120에서 SGLang flashinfer 미지원(CUDA 13.0 vs 12.9 mismatch) + vLLM EngineCore 360MiB 고정으로 10개만, torch.compile reduce-overhead 233MiB Peak 265MiB가 유일한 400MiB 이하 + 50개 해법 - 2달 실험으로 증명
- **int8 232→515MiB 역전이 면접 킬러:** "작은 모델 왜 int8 안쓰나? 124M은 스케일 테이블 오버헤드로 2배 증가, 7B 이상만 효과 - Day27 515MiB + quantized::linear_dynamic CPU only 로그 증명" 30초 답변 완성
- **QnA:** "2달 902→233 어떻게 했나? 375M→124M 227MB 절감 + len 32→8 75% KV 절감 + torch.compile cudagraphs=False로 Peak 844→265 68% 절감 + SGLang Blackwell Fail로 torch.compile 최종 + 50개 12.1GiB <16GiB Least VRAM LB - 5060 Ti 1대로 74% 절감 3배 증가 프로덕션 완성했습니다"

---

## 🇺🇸 English

### Measured Results

| Metric | Value | Meaning |
| --- | --- | --- |
| **Day0-24** | `902MiB 17 replicas` | Numpy 1-5 → 375M baseline, 2-month start |
| **Day25 vLLM** | `608MiB 10 replicas` | PagedAttention 360MiB overhead fail, only 10 |
| **Day26 580** | `580.8MiB / 844.3MiB Peak` | First torch.compile 24L 1024dim |
| **Day27 233** | `233.4MiB / 265.5MiB Peak` | 12L 768dim 68% Peak cut smashed 400MiB |
| **Day28 SGLang** | `fail CUDA 13.0 vs 12.9` | Blackwell sm_120 flashinfer not supported, torch.compile final |
| **Day29 50** | `233.4MiB / 445.5MiB / 12400MiB=12.1GiB OK` | 50 prod theory verified Least VRAM LB |
| **Day30 Final 2-month** | `233.4MiB / 265.5MiB Peak P0 20W` `12422MiB=12.1GiB <16GiB OK` `74% cut 17→50 3x` | Day24 902→233 74% cut final, Peak 265 lowest repro, 50 A/B prod possible |
| **int8 reverse** | `232→515MiB increase` | Small model int8 overhead proves bf16 floor |
| **After** | `34MiB P0 20W No running` | Full return |

**How 902→233 74% in 2 months? (PHASE 1→2)**

- PHASE 1 Day0-24: Numpy 1-5 start, 375M 757MB 902MiB baseline, python:3.12-slim torch cu128 sm_120 native
- PHASE 2 Day25: vLLM 608MiB 10 - PagedAttention EngineCore 360MiB overhead fail, less than 17
- PHASE 2 Day26-27: 375M→124M shrink 227MB + len 32→8 75% KV + torch.compile reduce-overhead fullgraph=False cudagraphs=False empty_cache() high precision 580/844→233/265 68% Peak cut
- Day28-29: SGLang fail mismatch Blackwell not supported + int8 232→515 increase bf16 floor + 50 12.1GiB <16GiB Least VRAM LB verified
- Day30 Final: 233.4/265.5 Peak repro 2-month lowest, 902→233 74% cut +17→50 3x = Meta prod portfolio done

### 💡 Key Insight

- **2-month final 233.4/265.5 Peak 74% cut 50 3x:** Day0 Numpy 1 → Day30 233MiB, 902MiB 17 →50 12.1GiB <16GiB 3x, Peak 844→265 68% cut stability - Blackwell optimization final with single 5060 Ti
- **SGLang Fail + 608 10 vs 233 50:** Blackwell sm_120 SGLang flashinfer not supported mismatch + vLLM 360MiB fixed 10 only, torch.compile 233 Peak 265 only <400MiB +50 solution - proven by 2-month exp
- **int8 232→515 reverse is interview killer:** "Why no int8 small? 124M scale table overhead 2x increase, only 7B+ benefits - Day27 515MiB + quantized::linear_dynamic CPU only proof" 30sec answer
- **QnA:** "How 902→233 in 2 months? 375M→124M 227MB + len 32→8 75% KV + torch.compile cudagraphs=False Peak 844→265 68% + SGLang Fail torch.compile final +50 12.1GiB <16GiB Least VRAM LB - 74% cut 3x prod with single 5060 Ti"

---

## 📊 Logs / 로그
```
2-Month Final 233MiB Peak 265MiB
final_2month.txt # Final Day30 233.4MiB / 265.5MiB Peak 74% cut 17→50 3x 12422MiB=12.1GiB <16GiB OK
nvidia_34MiB_return.txt # 34MiB P0 20W No running processes 41C full return

History / 히스토리
day24_902MiB_baseline.txt # 902MiB P8 5W 375M 757MB 17 replicas start
day25_vLLM_608MiB_10.txt # 608MiB PagedAttention EngineCore 360MiB overhead 10 only
day26_580MiB_844Peak.txt # 580.8MiB / 844.3MiB Peak 24L 1024dim first torch.compile
day27_233MiB_265Peak.txt # 233.4MiB / 265.5MiB Peak cudagraphs=False 74% cut smashed 400MiB
day28_sglang_fail.txt # CUDA 13.0 vs 12.9 mismatch Blackwell sm_120 not supported
day29_50_prod_12.1GiB.txt # 233.4MiB / 445.5MiB Peak / 12400MiB=12.1GiB OK Least VRAM LB
day30_233MiB_265Peak_final.txt # Final 233.4MiB / 265.5MiB Peak 2-month best

```



## 🔧 Stack / 스택

- Version / 버전: CUDA 13.2 Driver 595.84, torch cu128 sm_120 36 SMs (12,0) Blackwell, Python 3.12 ai-env meta-ai conda, SGLang flashinfer sm_120 not supported
- Base Image / 베이스 이미지: conda ai-env + torch.compile mode reduce-overhead fullgraph=False + torch._inductor.config.triton.cudagraphs=False + torch.set_float32_matmul_precision('high') + empty_cache() → 233.4MiB / 265.5MiB Peak final
- GPU: RTX 5060 Ti 16GB 16311MiB sm_120 Blackwell P0 20W 41C
- Model / 모델: SimpleGPT 124M d_model 768 n_layer 12 vocab 50257 len 8 bf16 150MB, 233MiB stable 265MiB peak lowest, 50 replicas 12.1GiB <16GiB

## 💡 Next / 다음

- Day31: 2달 포트폴리오 패키징 - ai-1year/README.md 최상단에 902MiB 17 vs 608MiB 10 vs 233MiB 50 비교표 + 면접 3문장 + day24-30 링크 모음, 싱가포르 경유 미국 PHASE 3 Llama 3.2 1B 233MiB 방식 도전
- Day31: 2-month portfolio packaging - ai-1year/README.md top 902MiB 17 vs 608MiB 10 vs 233MiB 50 table + interview 3 sentences + day24-30 links, PHASE 3 Llama 3.2 1B with 233MiB method
