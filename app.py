import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your trained model
model = load_model('model3.h5')

# Class labels (replace these with your actual classes from val_generator.class_indices)
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Streamlit UI
st.title("🍚 Rice Grain Classifier")
st.write("Upload an image of a rice grain and the model will classify it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # match model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted: **{class_names[class_index]}** with {confidence*100:.2f}% confidence")
