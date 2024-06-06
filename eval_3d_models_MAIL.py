import matplotlib.pyplot as plt
import pandas as pd

# Read the CSV file
df1 = pd.read_csv('data_area.csv')
df2 = pd.read_csv('data_area_2.csv')
df3 = pd.read_csv('data_area_3.csv')

# print(df.head())
plt.plot(df1['Centerline IDX'], df1['Area'])
plt.plot(df2['Centerline IDX'], df2['Area'])
plt.plot(df3['Centerline IDX'], df3['Area'])

plt.show()
