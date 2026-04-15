import os
import hashlib
import shutil
from PIL import Image
import logging
from sklearn.model_selection import train_test_split
from config_data import CLASS_MAPPINGS, DATA_DIR, PROCESSED_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_image_hash(image_path):
    """Calculate the MD5 hash of an image."""
    hasher = hashlib.md5()
    try:
        with open(image_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    except Exception as e:
        logging.error(f"Error hashing {image_path}: {e}")
        return None

def is_valid_image(image_path):
    """Check if the image is valid and can be opened by PIL."""
    try:
        with Image.open(image_path) as img:
            img.verify() # Verify that it is, in fact, an image
        return True
    except Exception as e:
        logging.warning(f"Invalid image found and discarded: {image_path}")
        return False

def sanitize_and_split():
    """
    1. Iterates over raw dataset.
    2. Identifies and drops corrupt or duplicate images.
    3. Splitting into Train (80%), Val (10%), Test (10%).
    4. Copies them to the processed directory structure.
    """
    for split in ['train', 'val', 'test']:
        for class_name in CLASS_MAPPINGS.keys():
            os.makedirs(os.path.join(PROCESSED_DIR, split, class_name), exist_ok=True)
            
    for class_name in CLASS_MAPPINGS.keys():
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            logging.error(f"Class directory not found: {class_dir}")
            continue
            
        logging.info(f"Processing class: {class_name}")
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]
        
        valid_images = []
        seen_hashes = set()
        
        # Sanitization
        for img_path in image_files:
            if not is_valid_image(img_path):
                continue
                
            img_hash = get_image_hash(img_path)
            if img_hash and img_hash not in seen_hashes:
                seen_hashes.add(img_hash)
                valid_images.append(img_path)
            else:
                logging.warning(f"Duplicate image found and discarded: {img_path}")
                
        logging.info(f"Valid and unique images for {class_name}: {len(valid_images)}")
        
        if len(valid_images) == 0:
            logging.warning(f"No valid images for {class_name}. Skipping split.")
            continue
            
        # Stratified Split (80 Train / 10 Val / 10 Test)
        try:
            train_val, test = train_test_split(valid_images, test_size=0.10, random_state=42)
            train, val = train_test_split(train_val, test_size=0.1111, random_state=42) # 0.1111 * 0.90 approx 0.10
        except ValueError as e:
            logging.error(f"Not enough images to split for {class_name}: {e}")
            continue
            
        # Copy to processed directories
        def copy_files(file_list, split_name):
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                dest = os.path.join(PROCESSED_DIR, split_name, class_name, file_name)
                shutil.copy2(file_path, dest)
                
        copy_files(train, 'train')
        copy_files(val, 'val')
        copy_files(test, 'test')
        
    logging.info("Sanitization and Split complete. Data ready in processed folder.")

if __name__ == "__main__":
    sanitize_and_split()
