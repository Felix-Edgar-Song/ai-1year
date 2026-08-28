import torch
from torch.utils.data import Dataset, DataLoader

# Fake dataset with 100 samples (like 100 web pages in web builder)
class MyDataset(Dataset):
    def __len__(self): return 100
    def __getitem__(self, idx):
        # Input: 768-dim vector (same as hidden size), Label: 0 or 1
        return torch.randn(768), torch.randint(0,2,(1,))

dataset = MyDataset()
# batch_size=2 means 2 samples per step, shuffle=True randomizes order each epoch
loader = DataLoader(dataset, batch_size=2, shuffle=True)

print(f"Total {len(dataset)} samples, batch=2 -> 1 epoch = {len(loader)} steps")

for step, (x,y) in enumerate(loader, 1):
    if step==1:
        # x shape [2,768] = [batch, hidden] -> exact input for yesterday's nn.Linear(768,768) 0.59M
        print(f"Step1: x={x.shape} [2,768] y={y.shape} <- input for yesterday's Linear!")
    if step==3: break

print(f"200 steps with batch=2 = 400 samples seen")
print(f"CUDA: {torch.cuda.memory_allocated()/1024**2:.2f}MB (DataLoader uses ~0 VRAM)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
