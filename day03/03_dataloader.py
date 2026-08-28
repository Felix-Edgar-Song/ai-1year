import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
   def __len__(self): return 100
   def __getitem__(self, idx):
       return torch.randn(768), torch.randint(0,2,(1,))

dataset = MyDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)
print(f"전체 {len(dataset)}개, batch=2면 1 epoch = {len(loader)} steps")

for step, (x,y) in enumerate(loader, 1):
   if step==1:
      print(f"step1: x={x.shape} [2,768] y={y.shape} <- 어제 Liner 입력!")
   if step==3:  break

print(f"200 steps = batch 2면 400개 봄")
print(f"CUDA: {torch.cuda.memory_allocated()/1024**2:.2f}MB (DataLoader는 VRAM 거의 0)")
print(f"GPU: {torch.cuda.get_device_name(0)}")
