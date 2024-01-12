import matplotlib.pyplot as plt

# Read data from the text file
file_path = 'workflow_processed_data_output/image_translations/alignement_translations.txt'  # Replace 'your_file.txt' with the actual file path
data = []

with open(file_path, 'r') as file:
    for line in file:
        values = [float(x) for x in line.split()]
        data.append(values)

# Extract columns
column_1 = [row[0] for row in data]
column_2 = [row[1] for row in data]
column_3 = [row[2] for row in data]
column_4 = [row[3] for row in data]

# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(8, 12))

# Plot the data
axs[0].plot(column_1, column_2, "o", color='b')
axs[0].set_title('Page vs Translation X')

axs[1].plot(column_1, column_3, "o", color='g')
axs[1].set_title('Page vs Translation Y')

axs[2].plot(column_1, column_4, "o", color='r')
axs[2].set_title('Page vs Rotation')

# Set common labels and show the plot
plt.xlabel('Column 1')
plt.tight_layout()
plt.show()
