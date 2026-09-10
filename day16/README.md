# Day16 - vLLM 15208MiB 15GB vs HF 2496MiB 2.5GB Real Measured 120ms vs 2755ms 22.9x Same Text

## 🇰🇷 한국어

### 결과 요약 - 100% 실측 근거
- **Idle**: `nvidia-smi 104MiB / 16311MiB` (Xorg 67MiB + gnome-shell 6MiB)
- **HF 실측**: `104MiB -> 2496MiB` 2.5GB - TinyLlama-1.1B 1.1B * bf16 2.2GB + KV Cache 0.3GB = 2.5GB on-demand 할당
- **vLLM 실측**: `104MiB -> 15208MiB` 15.2GB `VLLM::EngineCore 15096MiB` 14.7GB - `gpu_memory_utilization=0.9` 90% 선확보
- **텍스트**: 둘 다 `" a test message..."` 100% 동일 6->26 tokens 20 completion Real==Gen

### 실측 비교표

| 항목 | HF (실측) | vLLM (실측) | 결과 |
|---|---|---|---|
| GPU Mem | 104->2496MiB 2.5GB | 104->15208MiB 15.2GB EngineCore 15096MiB | vLLM 6.1배 더 많이 선확보 |
| Time | 2755ms cold | 120ms | 22.9x faster |
| Text | a test message... | a test message... | 100% Same 6->26 tokens |
| 할당 방식 | on-demand 2.5GB | 90% pre-alloc 15.2GB PagedAttention | Fig 6.2는 15GB 안에서 페이지 관리 |

### 책 연결

**Chapter 6 Section 6.1 Figure 6.2 PagedAttention**
- 정의: 15.2GB를 block_size=16 페이지로 쪼개 block table 관리, 파편화 0
- 내 실측: 104->2496MiB HF vs 104->15208MiB vLLM, 15208MiB 안에서 6->26 tokens 120ms 처리
- 왜 15.2GB나 잡나: `--gpu-memory-utilization 0.9` 기본값, `--gpu-memory-utilization 0.3` 주면 5GB로 줄일 수 있음

**Chapter 4 Fig 4.5 Causal + Fig 4.15 Generation Loop**
- causal mask 내부, /v1/completions max_tokens 20이 Fig 4.15 API 버전

## 🇺🇸 English

### Summary 100% Real Measured
- **Idle**: 104MiB / 16311MiB
- **HF Real**: 104->2496MiB 2.5GB on-demand
- **vLLM Real**: 104->15208MiB 15.2GB EngineCore 15096MiB 14.7GB 90% pre-alloc
- **Perf**: HF 2755ms vs vLLM 120ms 22.9x faster same text 6->26 tokens 100% Same

### Real Comparison Table

| Item | HF | vLLM | Result |
|---|---|---|---|
| GPU Mem Real | 104->2496MiB 2.5GB | 104->15208MiB 15.2GB EngineCore 15096MiB | vLLM 6.1x more pre-alloc |
| Time Real | 2755ms cold | 120ms | 22.9x faster |
| Text Real | a test message... | a test message... | 100% Same 6->26 tokens |
| Alloc Type | on-demand 2.5GB | 90% pre-alloc 15.2GB PagedAttention | Fig 6.2 inside 15GB |

## Log - Real Measured

```
Idle
Thu Sep 10 23:22:27 2026
104MiB / 16311MiB
Xorg 67MiB gnome-shell 6MiB

HF Real Measured by User
104MiB -> 2496MiB 2.5GB
HF 2755ms <s> Hello world, this is a test message.
5. Send a message to a specific user:
To send

vLLM Real Measured
Thu Sep 10 23:23:27 2026
15208MiB / 16311MiB
VLLM::EngineCore 15096MiB 14.7GB
curl 6->26 tokens 20 completion 120ms
{"id":"cmpl-bde126f3608c7f7b","choices":[{"text":" a test message.\n\n5. Send a message to a specific user:\n\nTo send"}],"usage":{"prompt_tokens":6,"total_tokens":26,"completion_tokens":20}}
```


## Files

- `vllm_log.txt` - 6->26 tokens 120ms 15208MiB
- `hf_log.txt` - 2755ms 2496MiB same text

## References

- Raschka, S. (2024). Build a Large Language Model (From Scratch). Manning.
    - Ch 6 Sec 6.1 Fig 6.2 PagedAttention 104->15208MiB vs 104->2496MiB real measured
    - Ch 4 Sec 4.4 Fig 4.5 Causal Mask
    - Ch 4 Sec 4.6 Fig 4.15 Generation Loop 20 tokens
