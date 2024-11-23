import os
import cv2

# Paths to images and labels
image_folder = '../../train/images'
label_folder = '../../train/labels'
output_folder = '../../cropped_images'

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all label files
labels = [f for f in os.listdir(label_folder) if f.endswith('.txt')]

for label_file in labels:
    # Construct paths to label file and corresponding image
    label_path = os.path.join(label_folder, label_file)
    image_path = os.path.join(image_folder, label_file.replace('.txt', '.jpg'))  # Adjust if your images are not .jpg

    # Read the image
    img = cv2.imread(image_path)
    height, width, _ = img.shape

    # Open and read the label file
    with open(label_path, 'r') as file:
        for label in file:
            label_info = label.strip().split()

            # Skip lines that don't have exactly 5 values (class + 4 coordinates)
            if len(label_info) != 5:
                continue  # Skip invalid lines

            # Unpack coordinates, skip class identifier
            _, x_center, y_center, box_width, box_height = map(float, label_info)

            # Convert normalized coordinates to pixel values
            x_min = int((x_center - box_width / 2) * width)
            y_min = int((y_center - box_height / 2) * height)
            x_max = int((x_center + box_width / 2) * width)
            y_max = int((y_center + box_height / 2) * height)

            # Ensure valid crop coordinates
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)

            # Check if the crop is valid (coordinates should not be inverted)
            if x_min >= x_max or y_min >= y_max:
                print(f"Invalid crop for {label_file}, skipping this bounding box.")
                continue  # Skip invalid crops

            # Crop the image
            cropped_img = img[y_min:y_max, x_min:x_max]

            # Ensure the cropped image is not empty before saving
            if cropped_img.size == 0:
                print(f"Empty crop for {label_file}, skipping.")
                continue  # Skip empty crops

            # Save the cropped image
            output_img_path = os.path.join(output_folder, f"{os.path.splitext(label_file)[0]}_crop.jpg")
            cv2.imwrite(output_img_path, cropped_img)
