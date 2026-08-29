# Day4 - Autograd backward() 768 grads

## 🇰🇷 한국어
- Param: 768 = 768x1 (Day2 590,592에서 간단 버전)
- Forward: pred 0.053, loss 0.4492 생성 - true 1.0과의 차이
- Backward: loss.backward() 1줄이 768개 그래디언트 생성, grad norm 23.5로 증명
- Update: weight -= 0.01*grad = 어제 200 steps 중 1 step을 손으로 구현
- CUDA: 64.01MB (모델 + grad + CUDA context), nvidia-smi 573MiB 중 일부
- 의미: 0.59M 파라미터가 어떻게 학습되는지 원리 증명

## 🇺🇸 English
- Param: 768 = 768x1 (simplified from Day2 590,592)
- Forward: pred 0.053, loss 0.4492 - difference from true 1.0
- Backward: loss.backward() creates 768 gradients, grad norm 23.5 proves it
- Update: weight -= 0.01*grad = 1 step of yesterday's 200 steps by hand
- CUDA: 64.01MB (model + grad + CUDA context), part of nvidia-smi 573MiB
- Insight: How 0.59M params are trained, proven by hand

## Log
