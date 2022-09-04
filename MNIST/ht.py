import torch
import numpy as np
import time
from functions import *

torch.manual_seed(50)

# input and weights
# digital A
M = 1
K = 256
N = 4


A = torch.ones(M,K)
print(A)
print('A: ' + str(A))

# analog B
B = torch.randn(K,N) + 1e-12
B = torch.abs(B)
B_signs = (B<0).type(torch.int)
print('B: ' + str(B))
print('B_signs: ' + str(B_signs))

# correct result
C = torch.matmul(A,B)
print(A.shape)
print(B.shape)
print(C.shape)

result, valor, e = cim(A.detach().to(device), B.detach().to(device), 1, 12, 8, 6, permutation = 'sorted', prints=True, perc=[60,14,10,7,5,3,1], num_sec=8, b_set = [12,12,12,12,12,12,12,3])
print(valor)
print(e)