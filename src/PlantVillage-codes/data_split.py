import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Define dataset main folder (Modify this to your local dataset folder path)
DATASET_DIR = "data\PlantVillage-data\color" 

# train, validation, and test split ratios
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# Create dataset split folders
BASE_SPLIT_DIR = "dataset_split"
TRAIN_DIR = os.path.join(BASE_SPLIT_DIR, "train") # dataset_split\train2
VALID_DIR = os.path.join(BASE_SPLIT_DIR, "val") 
TEST_DIR = os.path.join(BASE_SPLIT_DIR, "test")

# Create folders if they don’t exist
for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    os.makedirs(folder, exist_ok=True) # the code's current path.....\dataset_split\train2

# Prepare dataset: Read folder names as class names & split images
def prepare_dataset():
    class_names = os.listdir(DATASET_DIR)  # Get folder names (each folder is a class)
    print(f"🔍 Found classes: {class_names}")

    for class_name in class_names:
        class_path = os.path.join(DATASET_DIR, class_name)
        if not os.path.isdir(class_path):
            continue  # Skip non-directory files
        
        images = os.listdir(class_path)
        #print(images[0])
        random.shuffle(images)
        
        # Split images
        train_images, temp_images = train_test_split(images, test_size=(1 - TRAIN_RATIO), random_state=42)
        valid_images, test_images = train_test_split(temp_images, test_size=(TEST_RATIO / (VALID_RATIO + TEST_RATIO)), random_state=42)
        
        # Copy images to respective folders
        for subset, subset_images in zip([TRAIN_DIR, VALID_DIR, TEST_DIR], [train_images, valid_images, test_images]):
            subset_class_dir = os.path.join(subset, class_name) # dataset_split\train2\apple
            os.makedirs(subset_class_dir, exist_ok=True)
            
            for img in subset_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(subset_class_dir, img)
                shutil.copy(src, dst)

    print("✅ Dataset prepared and split into train, validation, and test sets!")


prepare_dataset()
