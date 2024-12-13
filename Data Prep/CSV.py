import os
import csv

# Define the path to your image folder
image_folder_path = r''  # Adjust path if needed

# Get the list of all image files in the folder
image_files = [f for f in os.listdir(image_folder_path) if f.endswith('.jpg')]

# Define the CSV file path to output the result
csv_file_path = r''  # Adjust path if needed

# Open the CSV file for writing
with open(csv_file_path, mode='w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)

    # Write the header row
    csv_writer.writerow(['filename', 'no_seatbelt', 'seatbelt'])

    # Write the image filenames with empty label columns
    for image in image_files:
        csv_writer.writerow([image, '', ''])

print(f"CSV file saved at: {csv_file_path}")
