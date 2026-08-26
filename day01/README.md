# Day1 - Tensor & bf16 50% Proof

- GPU: 5060 Ti 16GB (16311MiB) Driver 595.84 CUDA 13.2
- Result: fp32 64MB vs bf16 32MB = 50% saving
- Connection: 4060 6710MiB 7B serving -> bf16 덕분, 5060 Ti는 2배 여유
- Env: ai-env Python 3.12.3 (venv, not conda)
