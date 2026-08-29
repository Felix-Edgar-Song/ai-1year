import torch
import torch.nn as nn

torch.manual_seed(0)
hidden = 768
# Simple model : 768 -> 1, param count 768 (bias=false for simplicity)
model = nn.Linear(hidden, 1, bias=False)
print(f"Param: {sum(p.numel() for p in model.parameters())} = 768x1")

# Fake data: 1 sample [1,768] - from yesterday's [2,768] simplifiad to 1
x = torch.randn(1, hidden, device = 'cuda')
y_true = torch.tensor([[1,0]], device = 'cuda')

# Move model to GPU bf16 - same as day 2
model = model.to(device= 'cuda', dtype=torch.bfloat16)
x = x.to(dtype=torch.bfloat16)

# 1. Forward: prediction
y_pred = model(x)
loss = (y_pred - y_true).pow(2).mean()
print(f"Forward: pred {y_pred.item():.3f}, loss {loss.item():.4f}")

# 2. Backward this 1 line creates gradients for all 768 params
loss.backward()
print(f"Backward: grad norm {model.weight.grad.norm().item():.4f} - 768 grads created!")

# 3. Update: SGD 1 step - 1 of yesterdays's 200 steps
with torch.no_grad():
	model.weight -= 0.01 * model.weight.grad
print(f"Upadted: weight mean {model.weight.mean().item():.6f}")

print(f"CUDA: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
print(f"GPU: {torch.cuda.get_device_name(0)}")
 
