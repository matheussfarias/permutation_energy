import torch
import numpy as np
import time
from functions import *

torch.manual_seed(50)

# input and weights
# digital A
M = 316
K = 100
N = 10


A = torch.randn(M,K)
print(A)
shape_prev = A.shape
A = A.flatten()
for i in range(len(A)):
    if A[i]<0:
        A[i]=0

A = A.reshape(shape_prev)
print('A: ' + str(A))

# analog B
B = torch.randn(K,N)

B_signs = (B<0).type(torch.int)
print('B: ' + str(B))
print('B_signs: ' + str(B_signs))

# correct result
C = torch.matmul(A,B)
print(A.shape)
print(B.shape)
print(C.shape)
result, valor, e, active, s, noise, n_adcs = cim(A.detach().to(device), B.detach().to(device), 1, 12, 8, 12, permutation = 'random', prints=True, perc=(K/1)*np.ones(1), num_sec=2, b_set = (8*np.ones((100,8))).tolist(), opt=0, add_noise=0, noise_gain=0)
print(valor)
print(e)
print(active)
print(s)
print(noise)
print(n_adcs)
#perc=[60,14,10,7,5,2,1]
#perc=[60,16,8,6,3,2,2]
#b_set = (12*np.ones((100,8))).tolist()