import os
import random
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F

# Define paths
input_folder = ""  # Folder containing original images
output_folder = ""  # Folder to save augmented images
os.makedirs(output_folder, exist_ok=True)

# Define the number of augmented images required
total_images = 6000

# Define augmentation transformations
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    transforms.RandomVerticalFlip(p=0.5),
])

# Load all image paths
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
assert len(image_files) > 0, "No images found in the input folder."

# Calculate how many augmentations to apply to each image
images_needed = total_images - len(image_files)
if images_needed <= 0:
    print(f"Your dataset already has {len(image_files)} images or more.")
    exit()

# Augment images
counter = 0
while counter < images_needed:
    for img_path in image_files:
        if counter >= images_needed:
            break
        # Open the image
        image = Image.open(img_path).convert("RGB")

        # Apply augmentation
        augmented_image = augmentation_transforms(image)

        # Save the augmented image
        output_path = os.path.join(output_folder, f"augmented_{counter + 1}.jpg")
        augmented_image.save(output_path)

        counter += 1

print(f"Augmentation completed. {total_images} images are now in the folder.")
