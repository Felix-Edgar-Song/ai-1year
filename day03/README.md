# Day3 - DataLoader 100->50steps batch2 [2,768]

- 전체 100개, batch=2면 1 epoch = 50 steps
- x=torch.Size([2,768]) = 어제 nn.Linear(768,768) 0.59M 입력 형태 일치
- 200 steps = 400개 샘플 = 100개면 2 epoch, 10k면 0.04 epoch
- CUDA 0.00MB - DataLoader는 VRAM 0, 모델만 1.13MB
- GPU 5060 Ti 16GB
