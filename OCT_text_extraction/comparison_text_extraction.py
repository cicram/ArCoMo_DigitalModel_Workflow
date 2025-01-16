import pandas as pd
import matplotlib.pyplot as plt

# Read the first CSV file (Frame, Area, Mean Diameter, Min, Max)
file1_path = 'ArCoMo_Data/output_text_extraction/ArCoMo10001area_diam_values.csv'
#file1_path = 'ArCoMo_Data/output_text_extraction/ArCoMo10002area_diam_values.csv'
df1 = pd.read_csv(file1_path)

# Read the second CSV file (Area, Frame, MeanD, MinD, MaxD)
file2_path = 'ArCoMo_Data/ArCoMo10001/ArCoMo01_oct_stat_lad_dist_pre_pci.csv'
#file2_path = 'ArCoMo_Data/ArCoMo10002/ArCoMo01_oct_stat_lad_mid_pre_pci.csv'

df2 = pd.read_csv(file2_path)

# Ensure both dataframes have comparable column names
df1.columns = ['Frame', 'Area', 'Mean_Diameter', 'Min', 'Max']
df2.columns = ['Area2', 'Frame', 'Mean_Diameter2', 'Min2', 'Max2']

# Merge the two dataframes on the 'Frame' column
comparison_df = pd.merge(df1, df2, on='Frame')

# Create a comparison of each variable (Area, Mean Diameter, Min, Max)
comparison_df['Area_diff'] = comparison_df['Area'] - comparison_df['Area2']
comparison_df['Mean_Diameter_diff'] = comparison_df['Mean_Diameter'] - comparison_df['Mean_Diameter2']
comparison_df['Min_diff'] = comparison_df['Min'] - comparison_df['Min2']
comparison_df['Max_diff'] = comparison_df['Max'] - comparison_df['Max2']

# Display the differences for inspection
print("Comparison of differences between the two files:")
print(comparison_df[['Frame', 'Area_diff', 'Mean_Diameter_diff', 'Min_diff', 'Max_diff']])


# Plot for Area
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Frame'], comparison_df['Area'], label='Area (File 1)', marker='o')
plt.plot(comparison_df['Frame'], comparison_df['Area2'], label='Area (File 2)', marker='x')
plt.plot(comparison_df['Frame'], comparison_df['Area_diff'], label='Difference (Area)', linestyle='--', color='red')
plt.xlabel('Frame')
plt.ylabel('Area (mm²)')
plt.title('Comparison of Area between File 1 and File 2')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Mean Diameter
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Frame'], comparison_df['Mean_Diameter'], label='Mean Diameter (File 1)', marker='o')
plt.plot(comparison_df['Frame'], comparison_df['Mean_Diameter2'], label='Mean Diameter (File 2)', marker='x')
plt.plot(comparison_df['Frame'], comparison_df['Mean_Diameter_diff'], label='Difference (Mean Diameter)', linestyle='--', color='red')
plt.xlabel('Frame')
plt.ylabel('Mean Diameter (mm)')
plt.title('Comparison of Mean Diameter between File 1 and File 2')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Min Diameter
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Frame'], comparison_df['Min'], label='Min Diameter (File 1)', marker='o')
plt.plot(comparison_df['Frame'], comparison_df['Min2'], label='Min Diameter (File 2)', marker='x')
plt.plot(comparison_df['Frame'], comparison_df['Min_diff'], label='Difference (Min Diameter)', linestyle='--', color='red')
plt.xlabel('Frame')
plt.ylabel('Min Diameter (mm)')
plt.title('Comparison of Min Diameter between File 1 and File 2')
plt.legend()
plt.grid(True)
plt.show()

# Plot for Max Diameter
plt.figure(figsize=(10, 6))
plt.plot(comparison_df['Frame'], comparison_df['Max'], label='Max Diameter (File 1)', marker='o')
plt.plot(comparison_df['Frame'], comparison_df['Max2'], label='Max Diameter (File 2)', marker='x')
plt.plot(comparison_df['Frame'], comparison_df['Max_diff'], label='Difference (Max Diameter)', linestyle='--', color='red')
plt.xlabel('Frame')
plt.ylabel('Max Diameter (mm)')
plt.title('Comparison of Max Diameter between File 1 and File 2')
plt.legend()
plt.grid(True)
plt.show()