import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

# Load the trained model
model = load_model(r'c:\Users\ASUS\Desktop\python\project_files\smnist.h5')  # Replace with your model path

# Define the classes for sign language (replace with your classes)
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Streamlit app
st.title("Sign Language Prediction App")
background_image = r"""
<style>
    body {
        background-image: url('c:\Users\ASUS\Pictures\sign.jpg');  /* Replace 'your_image_url.jpg' with the URL or path to your image */
        background-size: cover;
    }
</style>
"""
st.markdown(background_image, unsafe_allow_html=True)
# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file,color_mode='grayscale',target_size=(28, 28))
    img_array = image.img_to_array(img)
    #img_array=img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Display the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)


    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    # Display prediction
    st.subheader("Prediction:")
    st.write(f"The predicted sign is: {predicted_class}:)")

# Instructions for users
st.sidebar.header("How to Use:")
st.sidebar.markdown(
    "1. Upload an image of a hand gesture representing a sign language letter.\n"
    "2. Click the 'Predict' button to see the predicted sign language letter."
)
