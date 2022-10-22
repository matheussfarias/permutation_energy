import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)
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
n_weights = 8000 # number of weights
bits = 8 # number of bits to represent all weights
stdev = 1 # standard deviation for the distribution
n_sections =8 # number of sections

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
#w = torch.sort(w).values
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

#binary.sort(key=sorting_area)
binary_area=[]
for i in range(len(w)):
    binary_area.append(binary[i][0])

binary_area = np.array(binary_area)

binary = binary_area

print(len(binary))
'''
sec= [[],[],[],[],[],[],[],[]]
prob = [[],[],[],[],[],[],[],[]]
sections=[[],[],[],[],[],[],[],[]]
s = np.zeros(bits)
zero=0
for i in range(len(binary)):
    for j in range(bits):
        if binary[i][j]!=0:
            sec[bits-j-1].append(binary[i])
            s[bits-j-1]+=1
            break
        if j==bits-1:
            zero=zero+1

print(zero)
for i in range(len(sec)):
    length = len(sec[i])
    sec[i] = np.sum(sec[i],axis=0)
    prob[i] = sec[i]/length
    
print(np.vstack(sec))
print(np.vstack(prob))
print(s)
# PLOTTING s_i x i
print("Final s_i x i: ")
plot = np.ones(bits)
plt.stem(np.arange(1,bits+1), plot)
plt.ylabel(r'$s_i$', size=20)
plt.xlabel(r'$i$', size=20)
plt.show()
print(np.sum(prob))
exit()

'''
# SECTIONING WEIGHTS
'''sections = [[],[],[],[],[],[],[],[]]
sections[0].append(binary[62:137])
acc = [137-62, 329-137, 697-329, 1455-697, 2881-1455, 5311-2881, 7580-5311, 8000-7580]
sections[1].append(binary[137:329])
sections[2].append(binary[329:697])
sections[3].append(binary[697:1455])
sections[4].append(binary[1455:2881])
sections[5].append(binary[2881:5311])
sections[6].append(binary[5311:7580])
sections[7].append(binary[7580:])
'''
sections = binary.reshape(n_sections, int(n_weights/n_sections), bits)
print("Sections before accumulating")
print(sections)
print(" ")
sections = np.sum(sections, axis=1)
'''for i in range(len(sections)):
    sections[i]=np.sum(sections[i],axis=1)
sections=np.vstack(sections)'''
print("Sections")
print(sections)
print(" ")

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
plt.ylabel(r'$s_i$', size=20)
plt.xlabel(r'$i$', size=20)
plt.show()
print(sections/(n_weights/n_sections))
print(np.sum(sections/(n_weights/n_sections)))
'''dens=[]
for i in range(len(sections)):
    dens.append(sections[i]/acc[i])
print(dens)
dens = np.array(dens)
dens = np.vstack(dens)
print(dens)
print(np.sum(dens))'''
exit()

b = np.arange(0,13)

y1 = 30*b-33
y2 = 40*b - 60
plt.clf()
plt.plot(b, y1, label = 'sorted')
plt.plot(b,y2, label = 'unsorted')
plt.ylabel('Cost')
plt.xlabel('Number of bits')
plt.legend()
plt.show()