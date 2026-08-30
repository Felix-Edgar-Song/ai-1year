import torch
import torch.nn as nn

torch.manual_seed(0)
hidden = 768

#Model: 768->1, 768params - same as Day4
model = nn.Linear(hidden, 1, bias=False)
print(f"Param: {sum(p.numel() for p in model.parameters())} = 768")

#Fake data: batch 2 [2,768] from Day3 DataLoader
x = torch.randn(2, hidden, device='cuda')
y_true = torch.tensor([[1.0],[0.0]], device='cuda')

#To GPU bf16 - Day1 50% saving
model_sgd = nn.Linear(hidden, 1, bias=False).to(device='cuda', dtype=torch.bfloat16)
model_adamw = nn.Linear(hidden, 1, bias=False).to(device='cuda', dtype=torch.bfloat16)
model_adamw.weight.data = model_sgd.weight.data.clone()

opt_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
opt_adamw = torch.optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=0.01)

x = x.to(dtype=torch.bfloat16)
y_true = y_true.to(dtype=torch.bfloat16)

print("Step | SGD loss | AdamW loss")
for step in range(1,11):
    # SGD
    loss1 = (model_sgd(x) - y_true).pow(2).mean()
    loss1.backward()
    opt_sgd.step(); opt_sgd.zero_grad()

    # AdamW
    loss2 = (model_adamw(x) - y_true).pow(2).mean()
    loss2.backward()
    opt_adamw.step(); opt_adamw.zero_grad()

    print(f"{step:2d} | {loss1.item():.4f} | {loss2.item():.4f}")

print(f"CUDA: {torch.cuda.memory_allocated()/2014**2:.2f}MB - AdamW m+v 2x more")
print(f"GPU: {torch.cuda.get_device_name(0)}")

