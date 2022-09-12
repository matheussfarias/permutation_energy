import matplotlib.pyplot as plt

#4, 3, 2
#energy = [2048, 16320, 2256, 1016]
energy = [2048, 16320, 4000, 3080]
#final_error = [20.3408,28.075,15.8525,16.85]
final_error = [329.08,334.6206,111.2059,87.2621]
sections = [ 0,  0,  1,  3,  5, 12,  7,  0,  4]
plt.title('Energy Comparison')
plt.bar(['1 section', '8-unsorted', '8-sorted', '8-sorted+grid'],energy, color = 'blue')
plt.show()
plt.clf()

plt.title('Total Error Comparison')
plt.bar(['1 section', '8-unsorted', '8-sorted', '8-sorted+grid'],final_error, color = 'blue')
plt.show()
plt.clf()

plt.title('s_i x i')
plt.bar(['s0','s1', 's2', 's3', 's4', 's5', 's6', 's7', 's8'],sections, color = 'blue')
plt.show()
plt.clf()


