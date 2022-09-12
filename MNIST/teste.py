
import torch
from functions import *

seed=50
torch.manual_seed(seed)
#torch.cuda.manual_seed(seed)
#torch.cuda.manual_seed_all(seed)

x = 0.11
print(convert_to_neg_bin(x,6))