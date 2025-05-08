import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import create_model
from preprocess import create_data_generators, load_and_preprocess_image, visualize_augmented_images

def train_model(epochs=50, batch_size=32, img_size=(128, 128)):
    """
    Train the brain tumor detection model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Target image size
    
    Returns:
        Trained model and training history
    """
    # Create data generators
    train_generator, validation_generator = create_data_generators(batch_size, img_size)
    
    # Create model
    model = create_model(input_shape=(*img_size, 3))
    
    # Print model summary
    model.summary()
    
    # Callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('brain_tumor_model.h5', save_best_only=True, monitor='val_accuracy')
    ]
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Save the model
    model.save('brain_tumor_model.h5')
    print("Model saved as 'brain_tumor_model.h5'")
    
    return model, history

def evaluate_model(model, validation_generator):
    """
    Evaluate the model and generate performance metrics
    
    Args:
        model: Trained Keras model
        validation_generator: Validation data generator
    """
    # Reset the generator
    validation_generator.reset()
    
    # Get predictions
    y_pred = model.predict(validation_generator, steps=len(validation_generator))
    y_pred_classes = (y_pred > 0.5).astype("int32")
    
    # Get true labels
    validation_generator.reset()
    y_true = validation_generator.classes
    
    # Classification report
    print("\nClassification Report:")
    class_labels = list(validation_generator.class_indices.keys())
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss
    
    Args:
        history: Training history from model.fit()
    """
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_sample_predictions(model, num_samples=5):
    """
    Visualize predictions on sample images
    
    Args:
        model: Trained model
        num_samples: Number of samples to visualize
    """
    # Find sample images from both classes
    tumor_dir = os.path.join('data', 'tumor')
    no_tumor_dir = os.path.join('data', 'no_tumor')
    
    tumor_files = [os.path.join(tumor_dir, f) for f in os.listdir(tumor_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    no_tumor_files = [os.path.join(no_tumor_dir, f) for f in os.listdir(no_tumor_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select random samples
    np.random.seed(42)
    sample_files = np.random.choice(tumor_files + no_tumor_files, min(num_samples, len(tumor_files) + len(no_tumor_files)), replace=False)
    
    plt.figure(figsize=(15, 4*len(sample_files)//3 + 4))
    
    for i, file_path in enumerate(sample_files):
        # Load and preprocess image
        img = load_and_preprocess_image(file_path)
        
        # Make prediction
        img_batch = np.expand_dims(img, 0)
        prediction = model.predict(img_batch)[0][0]
        predicted_class = "Tumor" if prediction > 0.5 else "No Tumor"
        actual_class = "Tumor" if 'tumor' in file_path.replace('\\', '/').split('/') else "No Tumor"
        
        # Display image with predictions
        plt.subplot(len(sample_files)//3 + 1, 3, i+1)
        plt.imshow(img)
        color = "green" if predicted_class == actual_class else "red"
        plt.title(f"Actual: {actual_class}\nPredicted: {predicted_class}\nConfidence: {prediction:.2f}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    plt.show()

if __name__ == "__main__":
    # Create directories if they don't exist
    os.makedirs(os.path.join('data', 'tumor'), exist_ok=True)
    os.makedirs(os.path.join('data', 'no_tumor'), exist_ok=True)
    
    print("Starting training process...")
    
    # First, check if we have sample data
    tumor_dir = os.path.join('data', 'tumor')
    no_tumor_dir = os.path.join('data', 'no_tumor')
    
    if len(os.listdir(tumor_dir)) == 0 or len(os.listdir(no_tumor_dir)) == 0:
        print("ERROR: No sample data found. Please add MRI images to the data/tumor and data/no_tumor folders.")
        print("Check the README.md for instructions on setting up the dataset.")
        exit(1)
    
    # Train the model
    model, history = train_model(epochs=20, batch_size=16)  # Reduced epochs for demonstration
    
    # Plot training history
    plot_training_history(history)
    
    # Create data generators for evaluation
    _, validation_generator = create_data_generators(batch_size=16)
    
    # Evaluate model
    evaluate_model(model, validation_generator)
    
    # Visualize sample predictions
    visualize_sample_predictions(model)
    
    print("Training and evaluation completed.")
    print("Model saved as 'brain_tumor_model.h5'")
    print("Run 'streamlit run app.py' to launch the web interface.")