import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Load and preprocess a single image
    
    Args:
        image_path: Path to the image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed image as numpy array
    """
    # Read image
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0,1]
    img = img / 255.0
    
    return img

def create_data_generators(batch_size=32, img_size=(128, 128)):
    """
    Create training and validation data generators with augmentation
    
    Args:
        batch_size: Batch size for training
        img_size: Target image size
        
    Returns:
        training_generator, validation_generator
    """
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 20% for validation
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator
    train_generator = train_datagen.flow_from_directory(
        'data',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True,
        classes=['no_tumor', 'tumor']
    )
    
    # Validation generator
    validation_generator = validation_datagen.flow_from_directory(
        'data',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False,
        classes=['no_tumor', 'tumor']
    )
    
    return train_generator, validation_generator

def visualize_augmented_images(image, num_augmented=5):
    """
    Visualize augmented versions of an image
    
    Args:
        image: Original image as numpy array
        num_augmented: Number of augmented images to generate
    """
    # Create an image data generator with augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Reshape image for data generator (adds batch dimension)
    img_batch = np.expand_dims(image, 0)
    
    # Create a plot
    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, num_augmented + 1, 1)
    plt.imshow(image)
    plt.title('Original')
    plt.axis('off')
    
    # Generate and plot augmented images
    aug_iter = datagen.flow(img_batch, batch_size=1)
    
    for i in range(num_augmented):
        aug_img = next(aug_iter)[0].astype('float32')
        plt.subplot(1, num_augmented + 1, i + 2)
        plt.imshow(aug_img)
        plt.title(f'Augmented {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('augmentation_examples.png')
    plt.show()