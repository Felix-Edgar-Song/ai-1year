import torch
import torch.nn as nn

torch.manual_seed(0)
hidden = 768

model_sgd = nn.Linear(hidden, 1, bias=False).to(device='cuda', dtype=torch.bfloat16)
model_adamw = nn.Linear(hidden, 1, bias=False).to(device='cuda', dtype=torch.bfloat16)
model_adamw.weight.data = model_sgd.weight.data.clone()

x = torch.randn(2, hidden, device='cuda', dtype=torch.bfloat16)
y_true = torch.tensor([[1.0],[0.0]], device='cuda', dtype=torch.bfloat16)

# SGD lr 0.01 -> 0.001로 낮춰서 발산 방지
opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.001)
opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=0.01)

print(f"Param: 768 | Fixed: SGD lr 0.001 vs AdamW lr 1e-3")
print("Step | SGD loss | AdamW loss")
for step in range(1, 11):
    loss1 = (model_sgd(x) - y_true).pow(2).mean()
    loss1.backward()
    opt_sgd.step(); opt_sgd.zero_grad()

    loss2 = (model_adamw(x) - y_true).pow(2).mean()
    loss2.backward()
    opt_adamw.step(); opt_adamw.zero_grad()

    print(f"{step:2d} | {loss1.item():.4f} | {loss2.item():.4f}")

print(f"CUDA: {torch.cuda.memory_allocated()/1024**2:.2f}MB")
print(f"GPU: {torch.cuda.get_device_name(0)}")
