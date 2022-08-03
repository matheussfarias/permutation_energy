import numpy as np

f=open('energy_mnist_10tol_sorted_25.txt','r')
lines = np.genfromtxt('energy_mnist_10tol_sorted_25.txt', delimiter=',')

reading=[]
for i in range(len(lines)):
    reading.append(lines[i][0])
print(len(lines))
energy_sorted = np.sum(reading)
print(energy_sorted)
f.close()

f=open('energy_mnist_10tol_random_25.txt','r')
lines = np.genfromtxt('energy_mnist_10tol_random_25.txt', delimiter=',')
print(len(lines))
reading=[]
for i in range(len(lines)):
    reading.append(lines[i][0])

energy_random = np.sum(reading)
print(energy_random)
f.close()

print(energy_sorted/energy_random)