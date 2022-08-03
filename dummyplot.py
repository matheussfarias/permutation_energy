import matplotlib.pyplot as plt

# 6,5,4 mnist
acc = [97.42, 96.87]
energy = [6931200000, 4043200000]

# 6,5,4 cifar dla
acc = [93.48, 91.78]
energy = [31457280000, 18350080000]

# 8,7,6 cifar dla
#acc = [94.86, 94.71]
#energy = [125829120000, 73400320000]

# 6,5,4 cifar resnet18
#acc = [94.55, 93.97]
#energy = [125829120000, 73400320000]



plt.title('Energy Comparison')
plt.bar(['unsorted', 'sorted'],energy, color = 'crimson')
plt.show()
plt.clf()

plt.title('Accuracy Comparison')
plt.bar(['unsorted', 'sorted'],acc, color = 'crimson')
plt.show()
plt.clf()


