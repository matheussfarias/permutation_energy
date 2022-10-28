import torch
import numpy as np

adcs = (torch.mul([8, 4, 5, 6, 6, 6, 6, 7],np.ones((100,8)))).tolist()
print(adcs)