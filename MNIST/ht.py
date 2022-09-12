import torch
import numpy as np
import time
from functions import *

torch.manual_seed(50)

# input and weights
# digital A
M = 100
K = 100
N = 100


A = torch.randn(M,K) + 1e-12
print(A)
print('A: ' + str(A))

# analog B
B = torch.randn(K,N) + 1e-12
#shape_prev = B.shape
#B = B.flatten()
#for i in range(len(B)):
#    if B[i]<0:
#        B[i]=0
#
#B = B.reshape(shape_prev)
#print(B)
B_signs = (B<0).type(torch.int)
print('B: ' + str(B))
print('B_signs: ' + str(B_signs))

# correct result
C = torch.matmul(A,B)
print(A.shape)
print(B.shape)
print(C.shape)
result, valor, e = cim(A.detach().to(device), B.detach().to(device), 1, 12, 12, 6, permutation = 'sorted', prints=True, perc=(100/100)*np.ones(99), num_sec=100, b_set = (12*np.ones(100)).tolist())
print(valor)
print(e)
#perc=[60,14,10,7,5,3,1]