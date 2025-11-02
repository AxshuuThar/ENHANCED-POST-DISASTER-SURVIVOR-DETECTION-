import os
import shutil
from sklearn.model_selection import train_test_split

# Define source directories (update paths based on your new dataset structure)
source_dirs = [
    r"c:\Users\ebins\MiniPro\data\Post-Disaster-Dataset-main\879real survivor",
    r"c:\Users\ebins\MiniPro\data\Post-Disaster-Dataset-main\112TinyPerson\20ManyPerson",
    r"c:\Users\ebins\MiniPro\data\Post-Disaster-Dataset-main\112TinyPerson\4PartPerson",
    r"c:\Users\ebins\MiniPro\data\Post-Disaster-Dataset-main\112TinyPerson\7PartTiny",
    r"c:\Users\ebins\MiniPro\data\Post-Disaster-Dataset-main\112TinyPerson\81other",
    r"c:\Users\ebins\MiniPro\data\C2A_Dataset\train\images",
    r"c:\Users\ebins\MiniPro\data\C2A_Dataset\val\images"
]

# Define target directories for YOLO format
image_train_dir = r"c:\Users\ebins\MiniPro\data\yolo_dataset\images\train"
image_val_dir = r"c:\Users\ebins\MiniPro\data\yolo_dataset\images\val"
label_train_dir = r"c:\Users\ebins\MiniPro\data\yolo_dataset\labels\train"
label_val_dir = r"c:\Users\ebins\MiniPro\data\yolo_dataset\labels\val"

# Create target directories if they don't exist
os.makedirs(image_train_dir, exist_ok=True)
os.makedirs(image_val_dir, exist_ok=True)
os.makedirs(label_train_dir, exist_ok=True)
os.makedirs(label_val_dir, exist_ok=True)

# Collect all image and label files recursively
images = []
labels = []

for source_dir in source_dirs:
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith((".jpg", ".png")):  # Collect image files
                images.append(os.path.join(root, file))
            elif file.endswith(".txt") and file != "classes.txt":  # Collect label files, excluding "classes.txt"
                labels.append(os.path.join(root, file))

# Debug: Check the collected files
print(f"Collected {len(images)} images and {len(labels)} labels.")

# Extract basenames (without extensions) to match images and labels
image_basenames = {os.path.splitext(os.path.basename(img))[0] for img in images}
label_basenames = {os.path.splitext(os.path.basename(lbl))[0] for lbl in labels}

# Identify unmatched files
unmatched_images = image_basenames - label_basenames
unmatched_labels = label_basenames - image_basenames

# Print mismatches for debugging
print(f"Unmatched images: {len(unmatched_images)} -> {unmatched_images}")
print(f"Unmatched labels: {len(unmatched_labels)} -> {unmatched_labels}")

# Filter out unmatched images and labels
images = [img for img in images if os.path.splitext(os.path.basename(img))[0] in label_basenames]
labels = [lbl for lbl in labels if os.path.splitext(os.path.basename(lbl))[0] in image_basenames]

# Verify that images and labels now match
assert len(images) == len(labels), "Images and labels still don't match after filtering!"

print(f"After filtering: {len(images)} images and {len(labels)} labels.")

# Split into train and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Move files to target directories
for img, lbl in zip(train_images, train_labels):
    shutil.copy(img, os.path.join(image_train_dir, os.path.basename(img)))
    shutil.copy(lbl, os.path.join(label_train_dir, os.path.basename(lbl)))

for img, lbl in zip(val_images, val_labels):
    shutil.copy(img, os.path.join(image_val_dir, os.path.basename(img)))
    shutil.copy(lbl, os.path.join(label_val_dir, os.path.basename(lbl)))

print(f"Training set: {len(train_images)} images and labels.")
print(f"Validation set: {len(val_images)} images and labels.")
print("Dataset preparation completed successfully!")
