import matplotlib.pyplot as plt

# Paths to your .txt files

path_axial_correction1 = 'ArCoMo_Data/ArCoMo1/output/ArCoMo1axial_angle_correctionImage.xyz'
path_axial_correction2 = 'ArCoMo_Data/ArCoMo1/output/ArCoMo1axial_angle_correctionICP.xyz'
path_axial_correction3 = 'ArCoMo_Data/ArCoMo1/output/ArCoMo1axial_angle_correctionOverlap.xyz'

# Function to read angles from a .txt file
def read_angles_from_file(file_path):
    with open(file_path, "r") as file:
        angles = [float(line.strip()) for line in file.readlines()]
    return angles

# Read angles from each file
angles1 = read_angles_from_file(path_axial_correction1)
angles2 = read_angles_from_file(path_axial_correction2)
angles3 = read_angles_from_file(path_axial_correction3)

# Plotting
plt.figure(figsize=(10, 6))  # Optional: Specify figure size
plt.plot(angles1, label="Image")
plt.plot(angles2, label="ICP")
plt.plot(angles3, label="Overlap")

# Adding plot details
plt.xlabel("OCT pullback image")  # X-axis label
plt.ylabel("Rotation angle [°]")  # Y-axis label
plt.legend()  # Display legend to identify each line
plt.title("Comparison of Axial Rotation Angles from Different Files")  # Optional: Add a title
plt.show()
