import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Fashion MNIST Classifier",
    page_icon="👕",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

model = load_model()

# -----------------------------
# Class Labels
# -----------------------------
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# -----------------------------
# Title
# -----------------------------
st.title("👕 Fashion MNIST Classifier")
st.write("Upload a clothing image and the CNN model will predict its category.")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    # -----------------------------
    # Display Image
    # -----------------------------
    image = Image.open(uploaded_file).convert('L')  # grayscale
    st.image(image, caption="Uploaded Image", width="stretch")

    # -----------------------------
    # Preprocess Image
    # -----------------------------
    img = image.resize((28, 28))
    img_array = np.array(img)

    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # -----------------------------
    # Prediction
    # -----------------------------
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # -----------------------------
    # Output Result
    # -----------------------------
    st.subheader("Prediction Result")
    st.success(f"**{class_names[predicted_class]}** ({confidence*100:.2f}% confidence)")

    # -----------------------------
    # Top 3 Predictions
    # -----------------------------
    st.subheader("Top 3 Predictions")
    top_3 = np.argsort(prediction[0])[-3:][::-1]

    for i in top_3:
        st.write(f"{class_names[i]} : {prediction[0][i]*100:.2f}%")

    # -----------------------------
    # Confidence Chart
    # -----------------------------
    st.subheader("Confidence Scores")
    st.bar_chart(prediction[0])
