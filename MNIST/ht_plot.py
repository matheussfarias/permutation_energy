import matplotlib.pyplot as plt

#4, 3, 2
energy = [2048, 16320, 2486, 1016]
final_error = [20.3408,28.075,29.63,16.85]

plt.title('Energy Comparison')
plt.bar(['1 section', '8-unsorted', '8-sorted', '8-sorted+grid'],energy, color = 'blue')
plt.show()
plt.clf()

plt.title('Total Error Comparison')
plt.bar(['1 section', '8-unsorted', '8-sorted', '8-sorted+grid'],final_error, color = 'blue')
plt.show()
plt.clf()


