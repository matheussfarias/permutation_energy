import numpy as np

x = np.random.randn(100)

delta_E = np.random.randn(1)
delta = delta_E/(1000*np.random.randn(100))

total = np.round(np.sum(x)/delta_E)*delta_E
partial=[]
for i in range(len(x)):
    partial.append(np.round(x[i]/delta[i])*delta[i])

partial = np.sum(partial)

error_total = (1/2)*(np.sum(x) - total)**2
error_partial = (1/2)*(np.sum(x) - partial)**2

print(error_total)
print(error_partial)