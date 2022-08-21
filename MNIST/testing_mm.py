import torch
import numpy as np
import time
from functions import *

torch.manual_seed(50)
perc= [68,27]
perc= [80,10]
# input and weights
# digital A
M = 316
K = 100
N = 10

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
grid_number=1000
best_err=10000

#for i in range(grid_number):
#    first = 100*np.random.rand()
#    perc = [first, (100-np.floor(first))*np.random.rand()]
#    result, valor, e = cim(A.detach().to(device), B.detach().to(device), 1, 12, 12, 4, permutation = 'sorted', prints=True, perc=perc)
#    print(e)
#    if valor[0]<best_err:
#        best_perc = perc
#        best_err = valor[0]



#print(best_perc)
#print(best_err)
result, valor, e = cim(A.detach().to(device), B.detach().to(device), 1, 12, 12, 12, permutation = 'sorted', prints=True, perc=[68,27], num_sec=3, b_set = None)
print(valor)