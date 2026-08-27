# Day2 - nn.Linear 0.59M vs LoRA 12K

- nn.Linear(768,768) = 590,592 = 0.59M (weight 589,824 + bias 768)
- LoRA r=8 = 12,288 = 12K = 2.08% -> 7B 540K 0.1093% same formula
- VRAM bf16 1.13MB = fp32 2.25MB 50% saving (Day1 proof reused)
- GPU 5060 Ti 768MiB / 16311MiB idle Driver 595.84
