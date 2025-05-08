import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import shutil

def create_sample_dataset(download_path="sample_data"):
    """
    Create a sample dataset structure for the project
    
    This is useful for users who don't have any MRI images to start with.
    It will organize a few sample images into the correct folder structure.
    
    Args:
        download_path: Path where sample data is downloaded
    """
    # Create directories
    os.makedirs(os.path.join('data', 'tumor'), exist_ok=True)
    os.makedirs(os.path.join('data', 'no_tumor'), exist_ok=True)
    
    # Copy sample images to appropriate folders
    # (In a real implementation, this would download or extract sample images)
    try:
        if os.path.exists(download_path):
            # Copy tumor images
            tumor_files = [f for f in os.listdir(os.path.join(download_path, 'tumor')) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for f in tumor_files:
                src = os.path.join(download_path, 'tumor', f)
                dst = os.path.join('data', 'tumor', f)
                shutil.copy(src, dst)
                
            # Copy no-tumor images
            no_tumor_files = [f for f in os.listdir(os.path.join(download_path, 'no_tumor')) if f.endswith(('.jpg', '.jpeg', '.png'))]
            for f in no_tumor_files:
                src = os.path.join(download_path, 'no_tumor', f)
                dst = os.path.join('data', 'no_tumor', f)
                shutil.copy(src, dst)
                
            print(f"Sample dataset created with {len(tumor_files)} tumor images and {len(no_tumor_files)} no-tumor images.")
        else:
            print(f"Sample data directory '{download_path}' not found.")
    except Exception as e:
        print(f"Error creating sample dataset: {e}")

def check_gpu_availability():
    """
    Check if GPU is available for training
    
    Returns:
        bool: True if GPU is available, False otherwise
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU is available: {len(gpus)} GPU(s) found.")
        # Print GPU details
        for gpu in gpus:
            print(f"  - {gpu}")
        return True
    else:
        print("No GPU available. Training will use CPU, which may be slower.")
        return False

def predict_and_explain(model, image_path, target_size=(128, 128)):
    """
    Make a prediction on an image and explain the result with a heatmap
    
    Args:
        model: Trained Keras model
        image_path: Path to the input image
        target_size: Target image size
        
    Returns:
        prediction: Prediction probability
        heatmap_img: Heatmap visualization
    """
    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img_array = img / 255.0