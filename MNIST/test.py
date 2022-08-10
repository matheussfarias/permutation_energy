import torch

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

q=12
result = torch.FloatTensor(convert_to_neg_bin(0.6249999, q))
powers = torch.FloatTensor([2**(-i) for i in range(0,q)])
convert = torch.multiply(result,powers)
print(torch.sum(convert))
