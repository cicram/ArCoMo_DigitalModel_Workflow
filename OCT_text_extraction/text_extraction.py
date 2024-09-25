import pytesseract
from PIL import Image
import os
import re
import csv
import numpy as np
import cv2

def clean_value(value):
    """
    Cleans and corrects OCR errors in the extracted numeric text.
    - Replaces semicolons with periods
    - Removes spaces between digits
    - Replaces common OCR errors with the correct characters
    - Ensures the number has two decimal places
    """
    value = value.replace(';', '.').replace(':', '.')
    value = value.replace('?', '').replace('_', '').replace(' ', '')
    
    # Ensure correct formatting (last two digits after the decimal point)
    # Example: 14.5 should be 1.45, 250.0 should be 25.00
    # Check if the value is a whole number without a decimal point and is too large
    if '.' not in value and len(value) > 2:
        value = value[:-2] + '.' + value[-2:]
    
    return value


def filter_color(input_image):
        # Create masks for the two colors
        color1 = (0, 255, 0)
        color2 = (170, 94, 10)
        lower_color1 = np.array([color1[2], color1[1], color1[0]], dtype=np.uint8)
        upper_color1 = np.array([color1[2], color1[1], color1[0]], dtype=np.uint8)
        # Little offset given to ensure color of line-dots are included
        lower_color2 = np.array([color2[2] - 1, color2[1] - 1, color2[0] - 1], dtype=np.uint8)
        upper_color2 = np.array([color2[2] + 1, color2[1] + 1, color2[0] + 1], dtype=np.uint8)

        mask1 = cv2.inRange(input_image, lower_color1, upper_color1)
        mask2 = cv2.inRange(input_image, lower_color2, upper_color2)

        # Combine the masks to create a binary mask
        binary_mask = cv2.bitwise_or(mask1, mask2)

        return binary_mask

def extract_text_from_image(img):
    # Crop the image to the area where the text is located
    cropped_img = img.crop(text_area)

    text = pytesseract.image_to_string(cropped_img)

    # Convert cropped image to NumPy array for color filtering
    if False:
        # Apply color filter
        binary_mask = filter_color(np.array(cropped_img.convert('RGB')))

        # Convert the filtered NumPy array back to a Pillow image
        filtered_img = Image.fromarray(binary_mask)

        # Use pytesseract to extract text from the filtered image
        text = pytesseract.image_to_string(binary_mask)
    

    return text.strip()

def extract_values_from_text(text):
    # Regular expressions to match the desired values
    area_match = re.search(r'Area:\s*([\d.?:;\s]+)mm', text)
    mean_diameter_match = re.search(r'Mean Diameter:\s*([\d.?:;\s]+)mm', text)
    min_match = re.search(r'Min:\s*([\d.?:;\s]+)mm', text)
    max_match = re.search(r'Max:\s*([\d.?:;\s]+)mm', text)
    pattern = r'^\d{1,2}\.\d{2}$'
    # Extract, clean, and store the values in the arrays
    if area_match:
        area_value = clean_value(area_match.group(1))
        if re.match(pattern, area_value):
            area_values.append(float(area_value))
        else: 
            area_values.append(np.nan)
    else:
        area_values.append(np.nan)
        print("Failed to extract Area")
        print(text)

    if mean_diameter_match:
        mean_diameter_value = clean_value(mean_diameter_match.group(1))
        if re.match(pattern, mean_diameter_value):
            mean_diameter_values.append(float(mean_diameter_value))
        else: 
            mean_diameter_values.append(np.nan)
    else: 
        mean_diameter_values.append(np.nan)
        print("Failed to extract Mean Diameter")
        print(text)

    if min_match:
        min_value = clean_value(min_match.group(1))
        if re.match(pattern, min_value):
            min_values.append(float(min_value))
        else: 
            min_values.append(np.nan)
    else: 
        min_values.append(np.nan)
        print("Failed to extract Min Diameter")
        print(text)

    if max_match:
        max_value = clean_value(max_match.group(1))
        if re.match(pattern, max_value):
            max_values.append(float(max_value))
        else: 
            max_values.append(np.nan)  
    else:    
        max_values.append(np.nan)
        print("Failed to extract Max Diameter")
        print(text)

def process_tiff(input_path, start_oct_frame, end_oct_frame):
    # Open the .tif file
    img = Image.open(input_path)

    frame = start_oct_frame
    while True:
        try:
            print(frame)
            # Seek to the correct frame
            img.seek(frame)
            text = extract_text_from_image(img)
            extract_values_from_text(text)
            extracted_texts.append((frame, text))
            frame += 1
            frames.append(frame)
            if frame > end_oct_frame:
                break
        except EOFError:
            # No more frames to process
            break
    
    return extracted_texts

def save_to_csv(frames, area_values, mean_diameter_values, min_values, max_values, output_file):
    # Define the header for the CSV file
    headers = ['Frame', 'Area (mm^2)', 'Mean Diameter (mm)', 'Min (mm)', 'Max (mm)']
    
    # Write the data to a CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        
        # Write each row of data
        for i in range(len(frames)):
            writer.writerow([frames[i], area_values[i], mean_diameter_values[i], min_values[i], max_values[i]])

def get_oct_frames_info(file_path):# Read the file
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('oct_start'):
                oct_start = int(line.split(':')[1].strip())
            elif line.startswith('oct_end'):
                oct_end = int(line.split(':')[1].strip())
            elif line.startswith('oct_registration'):
                oct_registration = int(line.split(':')[1].strip())
    return oct_start, oct_end

# 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
ArCoMo_numbers = [ 100, 200, 300, 500, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Coordinates of the text area (left, top, right, bottom)
text_area = (0, 0, 550, 200)  

for arcomo_nbr in ArCoMo_numbers:
    folder_path = "ArCoMo_Data/ArCoMo" + str(arcomo_nbr)
    image_path = folder_path + "/ArCoMo" + str(arcomo_nbr) + "_oct.tif"
    output_path_csv = "ArCoMo_Data/output_text_extraction/ArCoMo" + str(arcomo_nbr) + "area_diam_values.csv"
    file_path_oct_info = folder_path + "/ArCoMo" + str(arcomo_nbr) + "_oct_frames_info.txt"
    output_path_plot = "ArCoMo_Data/output_text_extraction/plots/ArCoMo" + str(arcomo_nbr) + "area_diam_values.png"
    
    start_frame, end_frame = get_oct_frames_info(file_path_oct_info)
    print(f"Processing folder: {folder_path}")
    # Arrays to store the extracted values
    extracted_texts = []
    area_values = []
    mean_diameter_values = []
    min_values = []
    max_values = []
    frames = []
    texts = process_tiff(image_path, start_frame, end_frame)
    save_to_csv(frames, area_values, mean_diameter_values, min_values, max_values, output_path_csv)


    # Print the extracted values
    print("Number of extracted values:")
    print("Area values:", len(area_values))
    print("Mean Diameter values:", len(mean_diameter_values))
    print("Min values:", len(min_values))
    print("Max values:", len(max_values))

    # Plot the extracted values
    import matplotlib.pyplot as plt

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the values
    ax.plot(frames, area_values, label='Area')
    ax.plot(frames, mean_diameter_values, label='Mean Diameter')
    ax.plot(frames, min_values, label='Min')
    ax.plot(frames, max_values, label='Max')

    # Add labels and title
    ax.set_xlabel('Frame')
    ax.set_ylabel('Value (mm)')
    ax.set_title(f"Extracted Values Over Frames Arcomo{arcomo_nbr}")

    # Add a legend
    ax.legend()

    # Save the figure
    plt.savefig(output_path_plot)
