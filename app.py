import streamlit as st
import numpy as np
import tensorflow as tf
import PIL.Image as Image
from model import load_model  # Load your model here

# Function to make predictions
def predict(image):
    # Preprocess the image (resize and normalize)
    image = np.array(image.resize((224, 224))) / 255.0
    image = image[np.newaxis, ...]

    # Make a prediction using your loaded model
    result = model.predict(image)

    # Find the label with the highest confidence
    predicted_label_index = np.argmax(result)

    return predicted_label_index

# Load your model
model = load_model()  # Define this function in your model.py

# Set up the Streamlit app
st.title("Image Classifier -by- Akhter Mohiuddin. Method used: Transfer learning using TensorFlow Hub")
st.write("Upload an image to classify its contents.")

# Image upload widget
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Predict the uploaded image
    predicted_label_index = predict(image)
    
    # Display the predicted label
    with open("ImageNetLabels.txt", "r") as f:
        image_labels = f.read().splitlines()
    
    predicted_label = image_labels[predicted_label_index]
    st.write(f"Prediction: {predicted_label}")
