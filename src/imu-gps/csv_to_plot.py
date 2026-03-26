import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('output.csv', usecols=['longitude', 'latitude'])
plt.scatter(x=df['longitude'], y=df['latitude'])
plt.show()