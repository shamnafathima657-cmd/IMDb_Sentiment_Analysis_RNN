import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("sentiment_rnn_model.keras")

model = load_model()

# -----------------------------
# Load IMDb Word Index
# -----------------------------
@st.cache_resource
def load_word_index():
    with open("word_index.pkl", "rb") as f:
        word_index = pickle.load(f)

    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3

    return word_index

word_index = load_word_index()

# -----------------------------
# Text Preprocessing
# -----------------------------
def preprocess(text):

    words = text.lower().split()

    sequence = [
        word_index.get(word, 2)
        for word in words
    ]

    padded = pad_sequences(
        [sequence],
        maxlen=200,
        padding="post"
    )

    return padded

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("🎬 IMDb Sentiment Analysis")

st.sidebar.info(
"""
### Project

This application predicts whether an IMDb movie review is **Positive** 😊 or **Negative** 😞 using a **Recurrent Neural Network (SimpleRNN)** built with TensorFlow/Keras.

**Model**

- Embedding Layer
- SimpleRNN
- Dropout
- Dense Output Layer

Dataset:
IMDb Movie Reviews
"""
)

st.sidebar.markdown("---")

st.sidebar.subheader("✨ Example Reviews")

if st.sidebar.button("Positive Example"):
    st.session_state.review = "This movie was amazing with brilliant acting and a wonderful story."

if st.sidebar.button("Negative Example"):
    st.session_state.review = "Worst movie ever. Waste of time and money."

# -----------------------------
# Main Title
# -----------------------------
st.title("🎬 Movie Review Sentiment Analyzer")

st.write(
"""
Enter any movie review below and click **Predict Sentiment**.
"""
)

# -----------------------------
# Input
# -----------------------------
review = st.text_area(
    "Movie Review",
    value=st.session_state.get("review", ""),
    height=200,
    placeholder="Type your review here..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")
    else:

        processed = preprocess(review)

        prediction = model.predict(processed, verbose=0)[0][0]

        positive = prediction
        negative = 1 - prediction

        st.markdown("---")

        col1, col2 = st.columns(2)

        if prediction >= 0.5:

            with col1:
                st.success("## 😊 Positive Review")

            with col2:
                st.metric(
                    "Confidence",
                    f"{positive*100:.2f}%"
                )

            st.progress(float(positive))

        else:

            with col1:
                st.error("## 😞 Negative Review")

            with col2:
                st.metric(
                    "Confidence",
                    f"{negative*100:.2f}%"
                )

            st.progress(float(negative))

        st.markdown("---")

        st.subheader("Prediction Probabilities")

        col1, col2 = st.columns(2)

        col1.metric("😊 Positive", f"{positive*100:.2f}%")
        col2.metric("😞 Negative", f"{negative*100:.2f}%")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.caption(
"""
Developed using **TensorFlow • Keras • Streamlit**

Dataset: IMDb Movie Reviews
"""
)