import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('data_area.csv')
df_2 = pd.read_csv('data_area_2.csv')

# Plot 'Centerline IDX' vs 'Area'
fig = plt.figure()
plt.plot(df['Centerline IDX'], df['Area'], marker='o', linestyle='-', color='b')
plt.plot(df_2['Centerline IDX'], df_2['Area'], marker='o', linestyle='-', color='r')
diff = abs(df['Area']- df_2['Area'])

plt.plot(df_2['Centerline IDX'], diff*3, marker='o', linestyle='-', color='g')

import numpy as np
print(f"Mean{np.mean(diff[500:800])}, SD: {np.std(diff[500:800])}")
plt.xlabel('Centerline IDX')
plt.ylabel('Area')
plt.title('Centerline IDX vs Area')
plt.grid(True)


# Load the CSV file
df = pd.read_csv('data_volume.csv')
df_2 = pd.read_csv('data_volume2.csv')

# Plot 'Centerline IDX' vs 'Area'
fig2 = plt.figure()
plt.plot(df['Centerline IDX'], df['Volume'], marker='o', linestyle='-', color='b')
plt.plot(df_2['Centerline IDX'], df_2['Volume'], marker='o', linestyle='-', color='r')
diff = abs(df['Volume']- df_2['Volume'])

plt.plot(df_2['Centerline IDX'], diff*3, marker='o', linestyle='-', color='g')

import numpy as np
print(f"Mean{np.mean(diff[500:800])}, SD: {np.std(diff[500:800])}")
plt.xlabel('Centerline IDX')
plt.ylabel('Volume')
plt.title('Centerline IDX vs Volume')
plt.grid(True)
plt.show()


# Plot 'Centerline IDX' vs 'Area'
fig3 = plt.figure()
plt.scatter( df['Volume'][600:800], df_2['Volume'][600:800], marker='o', color='b')

import numpy as np
print(f"Mean{np.mean(diff[500:800])}, SD: {np.std(diff[500:800])}")
plt.xlabel('Centerline IDX')
plt.ylabel('Volume')
plt.plot([0, max(df['Volume'][600:800])], [0, max(df_2['Volume'][600:800])])
plt.title('Centerline IDX vs Volume')
plt.grid(True)
plt.show()