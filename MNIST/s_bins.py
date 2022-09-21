import torch
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

def quantize(x,q):
    low = torch.min(x)
    x_shifted = (x-low).detach().clone()
    high = torch.max(x_shifted)
    x_shifted_scaled = x_shifted*(2**q-1)/high
    return torch.tensor(torch.floor(x_shifted_scaled+.5),  dtype=torch.int16), (low, high)

def convert_to_bin(x,q):
    x = x.item()
    result=[]
    for i in range(q-1):
        result.append(x%2)
        x=x//2
    result.reverse()
    return result

# DEFINING VARIABLES
n_weights = 256 # number of weights
bits = 8 # number of bits to represent all weights
stdev = 1 # standard deviation for the distribution

# CREATING NORMAL DISTRIBUTION (JUST NON-NEGATIVE VALUES)
w = []
m = torch.distributions.half_normal.HalfNormal(stdev)
for i in range(n_weights):
    w.append(m.sample())
w = torch.FloatTensor(w)

# CONVERTING THE VALUES TO INTEGERS AND SORTING THEM
w, _ = quantize(w, bits)
w = torch.sort(w).values
print(w)

# PLOTTING THE HISTOGRAM TO SEE THE DISTRIBUTION
counts, bins = np.histogram(w)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

# CONVERTING EACH INTEGER TO ITS BINARY REPRESENTATION
binary = []
for i in range(len(w)):
    binary.append(convert_to_bin(w[i], bits))
binary = np.array(binary)

# ASSIGNING EACH BITLINE TO A s_i
s = np.zeros(bits-1)
active = np.zeros(bits-1)

for i in range(n_weights):
    print(binary[i])
    for j in range(bits-1):
        if binary[i][j]!=0:
            s[bits-j-2]+=1
            break
    if (np.sum(binary[i]))!=0:
        active[np.sum(binary[i])-1] += 1
    print(s)
    print(active)


# PLOTTING s_i x i
print(s)
plt.stem(np.arange(1,bits), s)
plt.show()

# PLOTTING NUMBER OF ACTIVE ADCS PER COLUMN
print(active)
plt.stem(np.arange(1,bits), active)
plt.show()