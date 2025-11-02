import os
import shutil

# Paths for the C2A_Dataset and YOLO dataset
c2a_train_images_dir = "C:/Users/ebins/MiniPro/data/C2A_Dataset/new_dataset3/train/images"
c2a_train_labels_dir = "C:/Users/ebins/MiniPro/data/C2A_Dataset/new_dataset3/train/labels"
c2a_val_images_dir = "C:/Users/ebins/MiniPro/data/C2A_Dataset/new_dataset3/val/images"
c2a_val_labels_dir = "C:/Users/ebins/MiniPro/data/C2A_Dataset/new_dataset3/val/labels"

# Existing YOLO dataset directories
yolo_train_images_dir = "C:/Users/ebins/MiniPro/data/yolo_dataset/images/train"
yolo_train_labels_dir = "C:/Users/ebins/MiniPro/data/yolo_dataset/labels/train"
yolo_val_images_dir = "C:/Users/ebins/MiniPro/data/yolo_dataset/images/val"
yolo_val_labels_dir = "C:/Users/ebins/MiniPro/data/yolo_dataset/labels/val"

# Helper function to copy files from source to destination
def copy_files(src_images_dir, src_labels_dir, dest_images_dir, dest_labels_dir):
    # Get list of all images and labels
    image_files = [f for f in os.listdir(src_images_dir) if f.endswith('.jpg') or f.endswith('.png')]
    label_files = [f for f in os.listdir(src_labels_dir) if f.endswith('.txt')]
    
    # Copy image and label files to the YOLO dataset directory
    for image in image_files:
        src_image_path = os.path.join(src_images_dir, image)
        dest_image_path = os.path.join(dest_images_dir, image)
        shutil.copy(src_image_path, dest_image_path)
    
    for label in label_files:
        src_label_path = os.path.join(src_labels_dir, label)
        dest_label_path = os.path.join(dest_labels_dir, label)
        shutil.copy(src_label_path, dest_label_path)

# Copy training and validation images and labels from C2A_Dataset to YOLO dataset
copy_files(c2a_train_images_dir, c2a_train_labels_dir, yolo_train_images_dir, yolo_train_labels_dir)
copy_files(c2a_val_images_dir, c2a_val_labels_dir, yolo_val_images_dir, yolo_val_labels_dir)

print("C2A_Dataset images and labels successfully copied to YOLO dataset.")
