import numpy as np
import torch

torch.manual_seed(50)

def quantize(x,q):
    low = torch.min(x)
    x_shifted = (x-low)
    high = torch.max(x_shifted)
    x_shifted_scaled = x_shifted*(2**q-1)/high
    x_quantized = (torch.floor(x_shifted_scaled.detach().clone()+.5)).type(torch.int16)
    return x_quantized, (low, high)

def dequantize(x, extra_args, q):
    low, high = extra_args 
    x_shifted = x.type(torch.float32)*high/(2**q-1) 
    x = x_shifted + low
    return x 

def convert_to_bin(x,q):
    x = x.item()
    result=[]
    for i in range(q):
        result.append(x%2)
        x=x//2
    result.reverse()
    return result

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

w = torch.rand(5)
x = torch.rand(5)
q = 8

# quantizing weights (finding indeces)
w_quant, args = quantize(w,q)
print(w)
print(w_quant)

# dequantizing weights
w_quantized = dequantize(w_quant, args, q)
print(w_quantized)

# converting to bitline representation
w_bin = []
for i in range(len(w)):
    w_bin.append(convert_to_bin(w_quant[i], q))
w_bin = torch.FloatTensor(w_bin)

# calculating output of c = x.w with cim array
c = torch.matmul(x, w_bin)
powers = torch.FloatTensor([2**(q-i-1) for i in range(0,q)])
result = torch.matmul(powers,c)
result_final = dequantize(result, args, q)

# using neg power
w_quantized_neg = []
for i in range(len(w)):
    w_quantized_neg.append(convert_to_neg_bin(w[i], q))
w_quantized_neg = torch.FloatTensor(w_quantized_neg)
powers_q = torch.FloatTensor([2**(-i) for i in range(0,q)])
print(powers_q)
print(w_quantized_neg)
result_neg = torch.matmul(powers_q,torch.matmul(x,w_quantized_neg))

print(torch.matmul(x, w))
print(torch.matmul(x, w_quantized))
print(result_final)
print(result_neg)