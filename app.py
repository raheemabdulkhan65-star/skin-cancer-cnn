import numpy as np
import streamlit as st
from PIL import Image
from tensorflow import keras

IMG_SIZE = 128   # must match the notebook

st.set_page_config(page_title="Skin Cancer Detection (CNN)", page_icon="🔬", layout="centered")

@st.cache_resource
def load_model():
    return keras.models.load_model("cnn_model.keras")

model = load_model()

st.title("🔬 Skin Cancer Detection")
st.caption("Convolutional Neural Network (TensorFlow / Keras) deployed with Streamlit")
st.write("Upload a close-up photo of a skin mole. The CNN predicts whether it looks **Benign** or **Malignant**.")

with st.sidebar:
    st.header("About")
    st.write("Model: CNN with 4 convolution blocks")
    st.write("Task: Benign vs Malignant (binary)")
    st.write(f"Input size: {IMG_SIZE} x {IMG_SIZE} RGB")
    st.warning("Educational project only. This is NOT a medical diagnosis. Please consult a doctor.")

file = st.file_uploader("Choose a skin image", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    st.sidebar.subheader("Image details")
    st.sidebar.write(f"Original width: {img.size[0]} px")
    st.sidebar.write(f"Original height: {img.size[1]} px")

    if st.button("Predict Diagnosis", type="primary"):
        x = np.expand_dims(np.array(img.resize((IMG_SIZE, IMG_SIZE)), dtype="float32"), axis=0)
        p_malignant = float(model.predict(x, verbose=0)[0][0])
        p_benign = 1.0 - p_malignant

        st.subheader("Result & Confidence Analysis")
        if p_malignant >= 0.5:
            st.error(f"Result: Malignant (confidence {p_malignant*100:.1f}%)")
        else:
            st.success(f"Result: Benign (confidence {p_benign*100:.1f}%)")

        st.progress(p_benign, text=f"Benign: {p_benign*100:.1f}%")
        st.progress(p_malignant, text=f"Malignant: {p_malignant*100:.1f}%")
        st.caption("This result is only a computer estimate and must not replace a doctor's opinion.")
