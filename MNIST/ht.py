import torch
import numpy as np
import time
from functions import *

torch.manual_seed(50)

# input and weights
# digital A
M = 2
K = 4
N = 3


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
ta = time.perf_counter()
result, valor, e, active, s, noise, n_adcs = cim(A.detach().to(device), B.detach().to(device), 1, 12, 8, 12, permutation = 'random', prints=True, perc=(K/1)*np.ones(1), num_sec=2, b_set = torch.mul(torch.FloatTensor([8, 8, 8, 8, 8,8, 8, 8]),torch.ones((1,8))).to(device), opt=0, add_noise=0, noise_gain=0)
tb = time.perf_counter()
print(tb-ta)
print(valor)
print(e)
print(active)
print(s)
print(noise)
print(n_adcs)
#perc=[60,14,10,7,5,2,1]
#perc=[60,16,8,6,3,2,2]
#b_set = (12*np.ones((100,8))).tolist()
#(np.multiply(np.array([8, 4, 5, 6, 6, 6, 6, 7]),np.ones((100,8)))).tolist()
#(8*np.ones((100,8))).tolist()