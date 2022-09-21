import matplotlib.pyplot as plt
import numpy as np
#4, 3, 2
energy = [2048, 16320, 2256, 1016]
#energy = [2048, 16320, 4000, 3080]
final_error = [20.3408,28.075,15.8525,16.85]
#final_error = [329.08,334.6206,111.2059,87.2621]
#sections = [ 0,  0,  1,  3,  5, 12,  7,  0,  4]


# 1 section 8 columns 12 bits
#sections = [ 0,  0,  0,  0,  0,  0,  0,  0, 10]
# 8 sections 8 columns unsorted 12 bits
#sections = [ 0,  0,  0,  0,  0,  0,  1,  7, 72]
#active = [ 0,  0,  0,  0,  0,  0,  1, 21, 58]
# 8 sections 8 columns sorted 12 bits
#sections = [ 0,  0,  0,  0,  4,  6, 19, 26, 25]
#active = [ 0,  0,  0,  0,  6, 11, 19, 29, 15]
# 8 sections 8 columns sorted grid

#--
# 1 section 8 columns 8 bits
#sections = [ 0,  0,  0,  0,  0,  0,  0,  0, 10]
# 8 sections 8 columns 8 bits unsorted
#[ 0.  0.  0.  0.  0.  0.  1. 21. 58.]
#[ 0.  0.  0.  0.  0.  0.  1.  7. 72.]


#energy = [20480, 163584, 136192]
#error = [1365.8137, 796.3165, 877.7741]


#energy = [327680, 2617344, 2179072]
#error = [711.0753, 709.3235, 708.7942]

#plt.stem(np.arange(0,9),[ 861, 11,   17,   22,   20,   29,   23,   17,    0])
#plt.stem(np.arange(0,9),[ 833, 4,    9,   10,   23,   34,   50,   26,   11])
#plt.stem(np.arange(1,9),[  3,  5,  5, 28, 25, 23,  7,  0])
plt.stem(np.arange(1,9),[ 2,  3,  9, 15, 26, 28, 14,  0])
plt.show()
exit()
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


