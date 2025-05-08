import os
import numpy as np
import cv2
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tqdm import tqdm

def setup_user_dataset():
    """Setup user-provided dataset"""
    print("\n=== Using Your Own Dataset ===")
    
    # Create directories
    tumor_dir = os.path.join('data', 'tumor')
    no_tumor_dir = os.path.join('data', 'no_tumor')
    
    os.makedirs(tumor_dir, exist_ok=True)
    os.makedirs(no_tumor_dir, exist_ok=True)
    
    # Check if the directories already have images
    tumor_images = [f for f in os.listdir(tumor_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    no_tumor_images = [f for f in os.listdir(no_tumor_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if tumor_images and no_tumor_images:
        print(f"Found existing dataset with {len(tumor_images)} tumor images and {len(no_tumor_images)} non-tumor images.")
        choice = input("Do you want to use this existing dataset? (y/n): ")
        if choice.lower() == 'y':
            return True
    
    # Guide user to add their dataset
    print("\nTo use your own dataset:")
    print(f"1. Place brain MRI images with tumors in: {os.path.abspath(tumor_dir)}")
    print(f"2. Place brain MRI images without tumors in: {os.path.abspath(no_tumor_dir)}")
    print("\nPlease add your images to these folders now.")
    input("\nPress Enter once you've added your images...")
    
    # Check if images were added
    tumor_images = [f for f in os.listdir(tumor_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    no_tumor_images = [f for f in os.listdir(no_tumor_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not tumor_images or not no_tumor_images:
        print("\nWarning: One or both directories are still empty.")
        print("You need both tumor and non-tumor images for the model to work.")
        return False
    
    print(f"\nDataset setup complete with {len(tumor_images)} tumor images and {len(no_tumor_images)} non-tumor images.")
    return True

def download_sample_images():
    """Setup dataset for brain tumor detection"""
    print("Setting up the brain tumor dataset...")
    
    # Create data directories
    os.makedirs('data/tumor', exist_ok=True)
    os.makedirs('data/no_tumor', exist_ok=True)
    
    # Guide user to set up their own dataset
    return setup_user_dataset()

def normalize_images():
    """Normalize all images to a consistent format and size"""
    print("\nNormalizing images to consistent format and size...")
    
    # Define directories
    tumor_dir = os.path.join('data', 'tumor')
    no_tumor_dir = os.path.join('data', 'no_tumor')
    
    # Process each directory
    for directory in [tumor_dir, no_tumor_dir]:
        images = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        for img_name in tqdm(images, desc=f"Processing {os.path.basename(directory)} images"):
            img_path = os.path.join(directory, img_name)
            
            # Read and resize image
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Could not read image {img_path}. Skipping...")
                continue
                
            # Resize to 128x128
            img_resized = cv2.resize(img, (128, 128))
            
            # Save back with consistent format
            output_path = os.path.join(directory, os.path.splitext(img_name)[0] + '.png')
            cv2.imwrite(output_path, img_resized)
            
            # If the output path is different from the original path, delete the original
            if output_path != img_path:
                os.remove(img_path)

def augment_data():
    """Generate additional samples using data augmentation"""
    print("\nAugmenting data to create more samples...")
    
    # Define data directories
    tumor_dir = 'data/tumor/'
    no_tumor_dir = 'data/no_tumor/'
    
    # Define augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Function to augment images in a directory
    def augment_directory(directory, prefix, num_augmented=4):
        files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg')) and not f.startswith('aug_')]
        
        for file in tqdm(files, desc=f"Augmenting {os.path.basename(directory)} images"):
            # Load image
            img_path = os.path.join(directory, file)
            img = cv2.imread(img_path)
            
            # Skip if image couldn't be loaded
            if img is None:
                print(f"Warning: Could not read {img_path}. Skipping...")
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            img = img.reshape((1,) + img.shape)  # Reshape for datagen
            
            # Generate augmented images
            i = 0
            for batch in datagen.flow(img, batch_size=1):
                i += 1
                # Convert back to BGR for OpenCV
                augmented_img = cv2.cvtColor(batch[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(directory, f'aug_{i}_{file}'), augmented_img)
                if i >= num_augmented:
                    break
    
    # Augment tumor and non-tumor images
    augment_directory(tumor_dir, 'tumor')
    augment_directory(no_tumor_dir, 'no_tumor')
    
    # Count and display stats
    tumor_count = len(os.listdir(tumor_dir))
    no_tumor_count = len(os.listdir(no_tumor_dir))
    print(f"Total images after augmentation - Tumor: {tumor_count}, No Tumor: {no_tumor_count}")

def preview_dataset():
    """Show a preview of the dataset"""
    print("\nGenerating dataset preview...")
    
    # Create directories for train/test split (will be used by the training script)
    os.makedirs('data/train/tumor', exist_ok=True)
    os.makedirs('data/train/no_tumor', exist_ok=True)
    os.makedirs('data/test/tumor', exist_ok=True)
    os.makedirs('data/test/no_tumor', exist_ok=True)

    # Simple preview of images
    plt.figure(figsize=(10, 5))
    
    # Plot tumor samples
    tumor_samples = [f for f in os.listdir('data/tumor') if f.endswith(('.jpg', '.png', '.jpeg'))][:5]
    if tumor_samples:
        for i, img_name in enumerate(tumor_samples):
            img = cv2.imread(os.path.join('data/tumor', img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 5, i+1)
            plt.imshow(img)
            plt.title("Tumor")
            plt.axis('off')
    else:
        print("Warning: No tumor images found to preview")
    
    # Plot non-tumor samples
    no_tumor_samples = [f for f in os.listdir('data/no_tumor') if f.endswith(('.jpg', '.png', '.jpeg'))][:5]
    if no_tumor_samples:
        for i, img_name in enumerate(no_tumor_samples):
            img = cv2.imread(os.path.join('data/no_tumor', img_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 5, i+6)
            plt.imshow(img)
            plt.title("No Tumor")
            plt.axis('off')
    else:
        print("Warning: No non-tumor images found to preview")
    
    plt.tight_layout()
    plt.savefig('data_preview.png')
    print("Dataset preview saved as 'data_preview.png'")
    plt.close()

def split_dataset():
    """Split the dataset into train and test sets"""
    print("\nSplitting dataset into training and testing sets...")
    
    # Get all images
    tumor_images = [f for f in os.listdir('data/tumor') if f.endswith(('.jpg', '.png', '.jpeg'))]
    no_tumor_images = [f for f in os.listdir('data/no_tumor') if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not tumor_images or not no_tumor_images:
        print("Error: One or both classes have no images. Cannot split the dataset.")
        return
    
    # Shuffle images
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(tumor_images)
    np.random.shuffle(no_tumor_images)
    
    # Split ratio (80% train, 20% test)
    tumor_split = int(0.8 * len(tumor_images))
    no_tumor_split = int(0.8 * len(no_tumor_images))
    
    # Copy tumor images to train/test folders
    for i, img in enumerate(tumor_images):
        src = os.path.join('data/tumor', img)
        if i < tumor_split:
            dst = os.path.join('data/train/tumor', img)
        else:
            dst = os.path.join('data/test/tumor', img)
        shutil.copy(src, dst)
    
    # Copy no_tumor images to train/test folders
    for i, img in enumerate(no_tumor_images):
        src = os.path.join('data/no_tumor', img)
        if i < no_tumor_split:
            dst = os.path.join('data/train/no_tumor', img)
        else:
            dst = os.path.join('data/test/no_tumor', img)
        shutil.copy(src, dst)
    
    # Print stats
    print(f"Training set - Tumor: {len(os.listdir('data/train/tumor'))}, No Tumor: {len(os.listdir('data/train/no_tumor'))}")
    print(f"Testing set - Tumor: {len(os.listdir('data/test/tumor'))}, No Tumor: {len(os.listdir('data/test/no_tumor'))}")

if __name__ == "__main__":
    if download_sample_images():
        normalize_images()
        augment_data()
        preview_dataset()
        split_dataset()
        print("\nDataset setup completed successfully!")
    else:
        print("\nDataset setup failed. Please add images and try again.")