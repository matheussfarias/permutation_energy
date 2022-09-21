import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)
torch.manual_seed(42)
np.random.seed(42)

def quantize(x,q):
    low = torch.min(x)
    x_shifted = (x-low).detach().clone()
    high = torch.max(x_shifted)
    x_shifted_scaled = x_shifted*(2**q-1)/high
    return torch.tensor(torch.floor(x_shifted_scaled+.5),  dtype=torch.int16), (low, high)

def convert_to_bin(x,q):
    x = x.item()
    result=[]
    for i in range(q):
        result.append(x%2)
        x=x//2
    result.reverse()
    return result

def sorting_sum(A):
    return A[-1]

def sorting_area(A):
    return A[-3]

# DEFINING VARIABLES
n_weights = 256 # number of weights
bits = 8 # number of bits to represent all weights
stdev = 1 # standard deviation for the distribution
n_sections = n_weights # number of sections

# CREATING NORMAL DISTRIBUTION (JUST NON-NEGATIVE VALUES)
w = []
m = torch.distributions.half_normal.HalfNormal(stdev)
#m = torch.distributions.normal.Normal(0,stdev)
#m = torch.distributions.uniform.Uniform(0,1)
for i in range(n_weights):
    w.append(m.sample())
w = torch.FloatTensor(w)

# CONVERTING THE VALUES TO INTEGERS AND SORTING THEM
w, _ = quantize(w, bits)
w = torch.sort(w).values
print("Weight values:")
print(w)
print(" ")

# PLOTTING THE HISTOGRAM TO SEE THE DISTRIBUTION
counts, bins = np.histogram(w)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()

# CONVERTING EACH INTEGER TO ITS BINARY REPRESENTATION
binary = []
for i in range(len(w)):
    converted = convert_to_bin(w[i], bits)
    msb=0
    for j in range(bits):
        if converted[j]!=0:
            msb = bits-j
            break
    binary.append((converted,np.sum(converted), msb, msb+np.sum(converted)))

binary.sort(key=sorting_sum)
binary_area=[]
for i in range(len(w)):
    binary_area.append(binary[i][0])

binary_area = np.array(binary_area)

binary = binary_area


# SECTIONING WEIGHTS
sections = binary.reshape(n_sections, int(n_weights/n_sections), bits)
print("Sections before accumulating")
print(sections)
print(" ")
sections = np.sum(sections, axis=1)
print("Sections")
print(sections)
print(" ")

means=[]
number_of_ones=[]
for i in range(binary.shape[0]):
    means.append(np.mean(binary[i]))
    number_of_ones.append(np.sum(binary[i]))

print(means)
plt.clf()
plt.plot(means)
plt.show()

print(number_of_ones)
plt.clf()
plt.plot(number_of_ones)
plt.show()

w_new=[]
powers_q = torch.FloatTensor([2**(bits-i-1) for i in range(0,bits)])
for i in range(len(w)):
    w_new.append(torch.sum(powers_q*binary[i]))

w_new = torch.stack(w_new)
plt.clf()
plt.plot(w_new)
plt.show()
exit()
# ASSIGNING EACH SECTION TO A s_i
s = np.zeros(bits)

for i in range(n_sections):
    print("Section "+ str(i+1) + ": ")
    print(str(sections[i]))
    for j in range(bits):
        if sections[i][j]!=0:
            s[bits-j-1]+=1
            break
    print("Partial s_i x i: ")
    print(s)
    print(" ")
print(" ")

# PLOTTING s_i x i
print("Final s_i x i: ")
print(s)
plt.stem(np.arange(1,bits+1), s)
plt.show()
