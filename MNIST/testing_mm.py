import torch
import numpy as np
import time
from functions import *

torch.manual_seed(50)


# input and weights
# digital A
M = 2
K = 12
N = 3

length = 12
num_sections=int(K/length)
A = torch.randn(M,K)
A = A*(A>0) + 1e-12
print('A: ' + str(A))

# analog B
B = torch.randn(K,N) + 1e-12
B_signs = (B<0).type(torch.int)
print('B: ' + str(B))
print('B_signs: ' + str(B_signs))

# correct result
C = torch.matmul(A,B)
print(A.shape)
print(B.shape)
result, valor, e = cim(A.detach().to(device), B.detach().to(device), 1, 12, 12, 6, permutation = 'random', prints=True)
print(e)