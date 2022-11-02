import torch
import time
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

def convert_to_neg_bin(x,N):
    result=[]
    i=0
    if x>=1:
        x=x-1
        result.append(1)
    else:
        result.append(0)
    while(i<N-1):
        if x*2<1:
            result.append(0)
            x=x*2
        else:
            result.append(1)
            x=x*2 - 1
        i+=1
    return result

def ef_convert_to_neg_bin(x,N):
    result = []
    i=0
    g1 = (x>=1).type(torch.int8)
    result.append(g1)
    x = x-g1

    while(i<N-1):
        db = 2*x
        g1 = (db>=1).type(torch.int8)
        result.append(g1)
        x = db - g1
        i+=1
    result = torch.transpose(torch.stack(result), 1, 2)
    result = torch.transpose(result, 0, 2)
    return result

x = 2*torch.rand(1000,1000).to(device)
q=8

print(x)
ta = time.perf_counter()
result = ef_convert_to_neg_bin(x,q)
tb = time.perf_counter()
print(result)
print(tb-ta)
result2 = result

print(x)
ta = time.perf_counter()
result = []
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        result.append(convert_to_neg_bin(x[i][j], q))
result = torch.FloatTensor(result).reshape(x.shape[0],x.shape[1],q).to(device)
tb = time.perf_counter()
print(result)
print(tb-ta)
result1 = result

print((result1 == result2).all())
exit()

B_digital=[]
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        B_digital.append(convert_to_neg_bin(x[i][j],q))

B_digital = torch.FloatTensor(B_digital).reshape(x.shape[0],x.shape[1],q).to(device)
print(B_digital)