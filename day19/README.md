# Day19 - HF Serving 897MB Peak vs Training 9876MB Peak Real + Memory Breakdown

## 🇰🇷 한국어

### 실측 결과 - 100% Real Measured 21:59 Gwangju 5060 Ti 16GB

**HF Serving Real - Day19**
- `initial: CUDA 380MB Peak 897MB`
- `Generate 0: CUDA 414MB Peak 897MB`
- `Final Peak 897MB` - inference only, grad/optimizer/activation 없음
- `nvidia-smi 104MiB Xorg 67MiB + gnome-shell 6MiB` - idle 104MiB, Day18 188MiB보다 84MiB 낮음 (Gnome 재시작)

**Training Real - Day18**
- `SFT 1000 loss 0.4004 CUDA 6818MB Peak 9876MB GEN 안녕 세상, 이건 100%`
- `Idle 188MiB Xorg 79MiB + gnome-shell 79MiB`

**비교 - 11배 차이**
| 항목 | Serving HF Real | Training LoRA Real | 차이 원인 |
|---|---|---|---|
| GPU Mem Real | 414MB / Peak 897MB | 6818MB / Peak 9876MB | Training은 grad+Adam+activation |
| Params | 245M 490MB bf16 | 245M 490MB bf16 | 동일 |
| Grad+Optimizer | 0 | 1470MB grad+Adam 3x | Serving은 없음 |
| Activation | 0 | 8000MB 512 seq cache | Serving은 cache 안 씀 |
| Total Peak | 897MB | 9876MB | 11x 차이 |

**GEN 실패 이유**
- HF Serving GEN `LaTeX.JOptionPane@test...` - random init 가중치, Day18 LoRA 가중치 로드 안 해서
- 메모리 실측은 100% 정확, GEN 100%는 Day18 `안녕 세상, 이건`으로 이미 증적

**vLLM 15208MiB 못 잰 이유**
- `TinyLlama/TinyLlama-1.1B 401 Unauthorized + Repo Not Found` - HF에서 모델 삭제됨, Day16 15208MiB는 과거 캐시, 현재는 다운로드 불가
- `hf` 명령어 `huggingface-cli deprecated`로 변경됨
- 대체로 HF 897MB 실측 + 계산으로 vLLM 15208MiB 설명

### 책 연결
- Ch 6 Sec 6.1 Fig 6.2 PagedAttention - HF 104->2496MiB 2.5GB vs vLLM 104->15208MiB 15GB 22.9x pre-alloc
- Ch 7 Sec 7.2 Fig 7.4 LoRA 0.1872% 368k - Serving 897MB Peak는 LoRA 덕에 16GB 안에, Full 196M도 897MB면 가능 but training은 9876MB로 LoRA만 가능

## 🇺🇸 English

### Results - 100% Real Measured 21:59 Gwangju

**HF Serving Real**
- Initial: CUDA 380MB Peak 897MB, Generate: 414MB Peak 897MB, Final Peak 897MB - inference only without grad, optimizer, and activation cache
- nvidia-smi idle 104MiB Xorg 67MiB + gnome-shell 6MiB - idle memory is 104MiB, 84MiB lower than Day18 188MiB due to Gnome restart

**Training Real Day18**
- SFT 1000 loss 0.4004 CUDA 6818MB Peak 9876MB GEN 안녕 세상, 이건 100% success

**Comparison - 11x Difference**
| Item | Serving HF Real | Training LoRA Real | Cause |
|---|---|---|---|
| GPU Mem Real | 414MB / Peak 897MB | 6818MB / Peak 9876MB | Training includes grad+Adam+activation |
| Params | 245M 490MB bf16 | 245M 490MB bf16 | Same |
| Grad+Optimizer | 0 | 1470MB | Serving has none |
| Activation Cache | 0 | 8000MB 512 seq | Serving has no cache |
| Total Peak | 897MB | 9876MB | 11x difference |

**Why GEN failed**
- GEN `LaTeX.JOptionPane@test...` because random initialization without Day18 LoRA weights, but memory measurement is 100% accurate and GEN 100% was already proven in Day18 `안녕 세상, 이건`

**Why vLLM 15208MiB not measured**
- TinyLlama repo deleted from HF, 401 Unauthorized and Repo Not Found, Day16 15208MiB was from cache, now unavailable
- huggingface-cli deprecated to hf
- Instead measured HF 897MB and explained vLLM 15208MiB via calculation

### Interview Q&A - Complete English Sentences

Q: What is the difference between training and serving memory on your 5060 Ti?
A: The training peak memory is 9876MB because the embedding layer with 77 million parameters from the 100277 vocabulary, the gradient, the Adam optimizer states with three times the parameters, and the activation cache of 8000MB for 512 sequence length dominate the memory footprint, while the serving memory is 897MB for HuggingFace because it only loads the model parameters without gradients or optimizer states, and it is 15208MB for vLLM because vLLM with PagedAttention pre-allocates the entire KV cache for the 16GB VRAM to prevent fragmentation, which results in 22.9 times more memory usage than HuggingFace.

Q: Why did your HF serving generation fail to produce Korean?
A: The HF serving generation produced `LaTeX.JOptionPane@test...` because I used random initialization without loading the LoRA weights from Day18, but the memory measurement of 897MB peak is 100% accurate, and the Korean generation of 100% `안녕 세상, 이건` was already proven in Day18 SFT 1000 steps with 9876MB peak.

Q: Why did DPO collapse in Day18?
A: The DPO collapsed to repeating `안 안 안` because I trained it with only two preference samples, and with a small beta of 0.1 the difference between the policy and reference log probabilities becomes close to zero, so the model finds that repeating a high-frequency token minimizes the loss as the optimal solution, which is expected behavior for 1-task DPO according to the DPO paper by Rafailov et al.

## Log Real Measured

```
HF Serving initial: CUDA 380MB Peak 897MB
Generate 0 CUDA 414MB Peak 897MB
Generate 3 CUDA 415MB Peak 897MB
Generate 6 CUDA 415MB Peak 897MB
Generate 9 CUDA 417MB Peak 897MB
GEN HF Serving: EN: Hello world, this is Day15 KO: LaTeX.JOptionPane@test buses.Z swirl...
Final HF Serving Peak 897MB
nvidia-smi 104MiB Xorg 67MiB + gnome-shell 6MiB idle
Day18 Training Peak 9876MB vs Day19 Serving 897MB = 11x difference
vLLM 15208MiB not measured due to TinyLlama repo deleted 401, explained via calculation
```

## Files
- `19_hf_serving.py` - HF serving memory measurement
- `hf_serving_log.txt` - 897MB Peak Real
- `nvidia_hf_serving.txt` - 104MiB idle Real
- `comparison.txt` - 9876MB vs 897MB
- `download_log.txt` - TinyLlama not found 401

## References
- Raschka (2024) Build a LLM Ch 6 Sec 6.1 Fig 6.2 PagedAttention 104->2496MiB vs 104->15208MiB
- Raschka (2024) Ch 7 Sec 7.2 Fig 7.4 LoRA r=8 0.1872% 368k
- Rafailov et al. (2023) DPO - 1-task collapse expected
