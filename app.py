import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
)

def load_model():
    """Load the trained model"""
    model_path = 'brain_tumor_model.h5'
    if not os.path.exists(model_path):
        st.error("Model file not found! Please train the model first using 'python train.py'")
        st.stop()
    return tf.keras.models.load_model(model_path)

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess the uploaded image"""
    # Convert to numpy array
    img = np.array(image)
    
    # Convert from BGR to RGB if needed
    if img.shape[-1] >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values to [0,1]
    img = img / 255.0
    
    return img

def predict_tumor(model, img):
    """Make prediction on the image"""
    # Add batch dimension
    img_batch = np.expand_dims(img, axis=0)
    
    # Make prediction
    prediction = model.predict(img_batch)[0][0]
    
    # Return prediction probability and class
    return prediction, "Tumor" if prediction > 0.5 else "No Tumor"

def main():
    st.title("🧠 Brain Tumor Detection")
    st.write("Upload an MRI scan to detect if a brain tumor is present.")
    
    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a Convolutional Neural Network (CNN) to detect "
        "the presence of tumors in brain MRI scans. The model has been trained on a "
        "dataset of MRI scans with and without tumors."
    )
    
    st.sidebar.title("Instructions")
    st.sidebar.info(
        "1. Upload a brain MRI scan image\n"
        "2. Wait for the model to process the image\n"
        "3. View the prediction results\n"
        "4. The model will indicate whether a tumor is detected and provide a confidence score"
    )
    
    # Load model
    with st.spinner("Loading the brain tumor detection model..."):
        model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an MRI scan image...", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns([1, 1])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1.subheader("Uploaded MRI Scan")
        col1.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process image and make prediction
        with st.spinner("Analyzing the MRI scan..."):
            # Preprocess the image
            processed_img = preprocess_image(image)
            
            # Make prediction
            prediction_prob, prediction_class = predict_tumor(model, processed_img)
            
            # Display prediction
            col2.subheader("Prediction Result")
            
            # Define color based on prediction
            color = "red" if prediction_class == "Tumor" else "green"
            
            # Create prediction message
            col2.markdown(f"<h3 style='color: {color};'>Prediction: {prediction_class}</h3>", unsafe_allow_html=True)
            col2.markdown(f"<h4>Confidence: {prediction_prob:.2%}</h4>", unsafe_allow_html=True)
            
            # Create a progress bar for visualization
            col2.progress(float(prediction_prob))
            
            # Additional information based on prediction
            if prediction_class == "Tumor":
                col2.warning("⚠️ Tumor detected in the MRI scan. Please consult with a medical professional.")
            else:
                col2.success("✅ No tumor detected in the MRI scan.")
                
            col2.info("Note: This is an automated analysis and should not replace professional medical diagnosis.")

if __name__ == "__main__":
    main()