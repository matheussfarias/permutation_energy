import torch

seed=50
torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)

a = torch.randn(1)
print(a)
print('{0:.20f}'.format(a[0]))