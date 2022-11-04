import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

accuracy = [97.74, 97.54, 97.61, 76.79, 97.44, 16.25]
energy = [71800755968, 910807296, 17810117952, 228077312, 4377179152, 55266928]
bits = [8, 8, 6, 6, 4, 4]

data = {
    'Accuracy': [97.74, 97.54, 97.61, 76.79, 97.44, 16.25],
    'Energy': np.array([71800755968, 910807296, 17810117952, 228077312, 4377179152, 55266928])/71800755968,
    'Bits': [8, 8, 6, 6, 4, 4],
    'Section': ['full', 'one', 'full', 'one', 'full', 'one']
}
df = pd.DataFrame(data)
print(df)

sns.barplot(data=df, x="Bits", y="Accuracy", hue="Section", palette='Reds')
plt.title('Accuracy for different number of bits on MNIST')
plt.show()
plt.plot()
sns.barplot(data=df, x="Bits", y="Energy", hue="Section", palette='Reds')
plt.title('Energy for different number of bits on MNIST')
plt.show()
plt.plot()