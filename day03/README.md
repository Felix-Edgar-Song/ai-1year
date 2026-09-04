# Day3 - DataLoader 100->50steps batch2 [2,768]

## 🇰🇷 한국어
- 전체 100개, batch=2면 1 epoch = 50 steps
- x=torch.Size([2,768]) = 어제 nn.Linear(768,768) 0.59M 입력 형태와 정확히 일치
- 200 steps = 400개 샘플 = 100개 데이터셋이면 2 epoch, 10,000개면 0.04 epoch
- CUDA 0.00MB - DataLoader는 VRAM을 거의 안 쓰고, 모델만 1.13MB (bf16) 차지
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB, Driver 595.84
- 의미: 어제 200 steps 67초 학습은 전체 데이터의 4%만 보고 loss 4.56→0.0999를 만든 것

## 🇺🇸 English
- Total 100 samples, batch=2 → 1 epoch = 50 steps (100/2)
- x=torch.Size([2,768]) = exact input shape for yesterday's nn.Linear(768,768) 0.59M
- 200 steps = 400 samples = 2 epochs on 100 dataset, 0.04 epoch on 10,000 dataset
- CUDA 0.00MB - DataLoader uses ~0 VRAM, only model uses 1.13MB (bf16)
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB, Driver 595.84
- Insight: Yesterday's 200 steps 67s training saw only 4% of data to get loss 4.56→0.0999

### LOG
```log
Total Samples: 100, batch=2 , 1 epoch = 50 steps
step1: x=torch.Size() y=torch.Size()
CUDA: 0.00MB (DataLoader's VRAM near -> 0)
GPU: NVIDIA GeForce RTX 5060 Ti[2][768][1]

