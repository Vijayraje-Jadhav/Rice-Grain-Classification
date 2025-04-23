import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image
import gdown

# File download setup
model_path = "model3.h5"
file_id = "1GhA1pjQtfoGY4YduqPTZs6BrbUFolkSY"
gdrive_url = f"https://drive.google.com/uc?id={file_id}"

# Download model only if not already present
if not os.path.exists(model_path):
    with st.spinner("Downloading model..."):
        gdown.download(gdrive_url, model_path, quiet=False)

# Load your trained model
model = load_model(model_path)

# Class labels
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Streamlit UI
st.title("🍚 Rice Grain Classifier")
st.write("Upload an image of a rice grain and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Adjust based on model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted: **{class_names[class_index]}** with {confidence*100:.2f}% confidence")
