<<<<<<< HEAD
# Brain Tumor Detection using Deep Learning

This project implements a Convolutional Neural Network (CNN) to detect brain tumors from MRI scans.

## Project Overview

Brain tumors are abnormal growths in the brain that can be benign or malignant. Early detection is crucial for effective treatment. This project uses deep learning to classify MRI scans into two categories:
- No Tumor: Normal brain MRI
- Tumor: Brain MRI with tumor present

## Dataset

The dataset consists of MRI brain scans in two categories:
- `data/no_tumor/`: MRI scans of normal brains
- `data/tumor/`: MRI scans of brains with tumors

## Setup Instructions

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place your MRI scan images in the appropriate folders or use the provided sample images

## Usage

### Training the Model

Run the training script:
```
python train.py
```

This will:
- Load the dataset
- Preprocess the images
- Train the CNN model
- Save the trained model as `brain_tumor_model.h5`
- Generate evaluation metrics and visualizations

### Using the GUI

Launch the Streamlit app:
```
streamlit run app.py
```

This will open a web interface where you can:
- Upload MRI scan images
- Get predictions on whether the scan shows a tumor
- View the confidence score of the prediction

## Model Architecture

The model uses a simple CNN architecture:
- Convolutional layers with ReLU activation
=======
# Brain-tumor-detection-using-cnn
>>>>>>> 219810effde8440311d369768d531fe5a2814386
