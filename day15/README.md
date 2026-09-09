# Day15 - BPE Real English 100% Match 0.0003 1295MB

## 🇰🇷 한국어

### 결과 요약
- **모델**: Param ~114M (vocab 100277*768=77M + 6 layers 37M), Fig 4.3 same + Fig 5.8 BPE cl100k_base
- **학습**: 100 steps 9.x -> 0.0003 (99.99%) 1295MB Peak 2052MB 25 steps 0.0007 4배 빠름
- **생성**: Prompt "Hello world, this is" -> Gen "Day15 causal GPT with real BPE tokenizer" 100% Match
- **어제 vs 오늘**: Day14 54.82M vocab 8k 494MB/673MB 0.0016 -> Day15 114M vocab 100k 1295MB/2052MB 0.0003 메모리 2.6배 증가는 vocab 때문
- **환경**: Ubuntu native ai-env, RTX 5060 Ti 16GB, 595.84 CUDA 13.2, Torch 2.11.0+cu128, tiktoken cl100k_base

### 책 연결

**Chapter 5 Section 5.2 Figure 5.8 BPE Tokenizer (오늘 핵심)**
- 정의: Byte Pair Encoding - "Hello"를 1개 토큰 9906이 아니라 빈도 높은 쌍 합치기, " world" 1917, " Day15" 6187처럼 의미 단위
- cl100k_base: GPT-4 토크나이저 100277 vocab, 영어 16개 토큰으로 압축 vs 문자 단위 60개
- 내 코드 `enc = tiktoken.get_encoding("cl100k_base")`가 이 그림 - `ids len=16`이 Fig 5.8 압축 증적
- 왜 필요한가: vocab 8k torch.randint는 가짜 숫자 외우기, 100k BPE는 진짜 영어 외우기 + vLLM serve시 동일 토크나이저 필수

**Chapter 4 Section 4.4 Figure 4.5 Causal Mask (어제 이어 유지)**
- 정의: `triu(diagonal=1) = -inf` 미래 가림, 오늘 코드 `seq_len = x.size(1)` 동적 마스크로 수정 - Fig 4.5가 고정 512가 아니라 입력 길이만큼 동적이어야 함이 핵심 교훈
- 결과: 512 고정 마스크는 319 입력에서 RuntimeError, 동적 마스크로 100% Match 해결

**Chapter 4 Section 4.6 Figure 4.15 Autoregressive Generation (오늘 증적)**
- 정의: `argmax` greedy로 10개 생성, Prompt "Hello world, this is" 5개 토큰 보고 뒤 10개 `[6187,868,59557,480,2898,449,1972,426,1777,47058]` 정확히 예측
- Real `is Day15 causal GPT with real BPE tokenizer.` == Gen 동일 100%

### 배운 점
- vocab 8k->100k 시 embed 메모리 6.1M->77M 12배, CUDA 494MB->1295MB 2.6배 - embedding이 메모리 대세
- causal mask는 고정 크기 512 쓰면 안 되고 `x.size(1)` 동적으로 해야 길이가 바뀌어도 동작
- BPE는 loss 0.0016->0.0003 더 낮게 만듦 - 영어 패턴이 숫자 랜덤보다 외우기 쉬움

## 🇺🇸 English

### Summary
- **Model**: Param ~114M (100k vocab), Fig 4.5 causal + Fig 5.8 BPE cl100k_base 100277 vocab
- **Training**: 100 steps -> 0.0003 99.99% 1295MB Peak 2052MB 25 steps 0.0007
- **Generation**: Prompt "Hello world, this is" -> "Day15 causal GPT with real BPE tokenizer" 100% Match

### Book Link
- Ch 5 Sec 5.2 Fig 5.8 BPE Tokenizer cl100k_base 100k vocab 16 tokens len
- Ch 4 Sec 4.4 Fig 4.5 Causal Mask dynamic `x.size(1)` fix 512 vs 319 error
- Ch 4 Sec 4.6 Fig 4.15 Generation Loop 10 tokens 100% Match

### Lesson
- Vocab 8k->100k embed 6.1M->77M memory 494MB->1295MB
- Causal mask must be dynamic not fixed 512
- BPE easier to memorize than randint 0.0016->0.0003

## Log
```
Text: Hello world, this is Day15 causal GPT with real BPE tokenizer.
BPE ids: len=16
x shape y shape
25 | loss 0.0007 | CUDA 1295MB | Peak 2052MB
50 | loss 0.0004 | CUDA 1295MB | Peak 2052MB
75 | loss 0.0004 | CUDA 1295MB | Peak 2052MB
100 | loss 0.0003 | CUDA 1295MB | Peak 2052MB
Prompt: Hello world, this is
Gen: Hello world, this is Day15 causal GPT with real BPE tokenizer
Gen ids tail:
Expected: is Day15 causal GPT with real BPE tokenizer.[9906][1917][11][420][374][6187][868][59557][480][2898][449][1972][426][1777][47058][13][1][512]
```

## Files

- `15_bpe_gpt.py` - English comments only, Fig 4.5 dynamic mask + Fig 5.8 BPE
- `log.txt` - 0.0003 100% Match 1295MB/2052MB

## References

- Raschka, S. (2024). Build a Large Language Model (From Scratch). Manning.
    - Chapter 5 Section 5.2 Figure 5.8 - BPE Tokenizer cl100k_base 100k vocab
    - Chapter 4 Section 4.4 Figure 4.5 - Causal Mask dynamic size
    - Chapter 4 Section 4.6 Figure 4.15 - Autoregressive Generation Loop
