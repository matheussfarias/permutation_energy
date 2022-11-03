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

x = 2*torch.rand(5,2).to(device)
a = torch.rand(3,5).to(device)
c = torch.matmul(a,x)
print(a)
print(x)
print(c)
q=8

#A = torch.tensor([1, 2, 3, 4])
#indices = torch.tensor([1, 0, 3, 2])
#result = torch.tensor([0, 0, 0, 0])
#print(result.scatter_(0, indices, A))

result = ef_convert_to_neg_bin(x,q)

#sorting mag
print(result)
shape = result.shape
stacked = result.reshape((shape[0]*shape[1], shape[2]))
powers = torch.FloatTensor([2**(-i) for i in range(-1,q-1)]).to('cuda')
indeces = torch.argsort(torch.sum(torch.mul(powers,stacked), axis=1))
sorted_stacked = stacked[indeces]
sorted_array = sorted_stacked.reshape(shape)
print(sorted_array)
exit()
print(torch.sum(stacked, axis=1))

exit()
val, indeces = torch.sort()
print(result)
print(result.shape)
exit()