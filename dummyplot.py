import matplotlib.pyplot as plt

#4, 3, 2
energy = [288, 168]
adc_error = [0.115, 0.1732]
final_error = [0.3846, 0.2254]

#6, 5, 4
energy = [1152, 672]
adc_error = [0.0101, 0.0105]
final_error = [0.0100, 0.0148]

plt.title('Energy Comparison')
plt.bar(['unsorted', 'sorted'],energy, color = 'crimson')
plt.show()
plt.clf()

plt.title('ADC Error Comparison')
plt.bar(['unsorted', 'sorted'],adc_error, color = 'crimson')
plt.show()
plt.clf()

plt.title('Total Error Comparison')
plt.bar(['unsorted', 'sorted'],final_error, color = 'crimson')
plt.show()
plt.clf()


