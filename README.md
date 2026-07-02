# IMDb_Sentiment_Analysis_RNN
A Deep Learning project that classifies IMDb movie reviews as Positive or Negative using a Recurrent Neural Network (SimpleRNN) built with TensorFlow/Keras and deployed with Streamlit.

#  IMDb Sentiment Analysis using Recurrent Neural Networks (RNN)

A Deep Learning project that automatically classifies IMDb movie reviews as **Positive** or **Negative** using a **Recurrent Neural Network (SimpleRNN)** built with **TensorFlow/Keras**. The project also includes an interactive **Streamlit web application** for real-time sentiment prediction.

---

##  Project Overview

Sentiment Analysis is a Natural Language Processing (NLP) task used to determine the emotional tone of text.

In this project, a SimpleRNN model learns the sequential relationships between words in IMDb movie reviews and predicts whether a review expresses a positive or negative sentiment.

---

##  Objectives

- Build a Recurrent Neural Network (SimpleRNN) for sentiment analysis.
- Classify IMDb movie reviews as Positive or Negative.
- Evaluate model performance using classification metrics.
- Deploy the trained model using Streamlit.
- Provide an interactive interface for real-time predictions.

---

##  Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Streamlit
- Pickle

---

##  Dataset

Dataset: **IMDb Movie Review Dataset**

- 50,000 movie reviews
- 25,000 Training Samples
- 25,000 Testing Samples
- Binary Classification
- Positive and Negative Reviews

Dataset Source:

https://keras.io/api/datasets/imdb/

---

##  Deep Learning Architecture

Embedding Layer

↓

SimpleRNN Layer

↓

Dropout Layer

↓

Dense Output Layer (Sigmoid)

---

##  Model Evaluation

The model is evaluated using:

- Accuracy
- Loss
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

## Project Structure

```
IMDb_Sentiment_Analysis_RNN
│
├── app.py
├── sentiment_rnn_model.keras
├── word_index.pkl
├── requirements.txt
├── README.md
```

---

##  Streamlit Application

Features:

- Predict movie review sentiment
- Interactive user interface
- Confidence score
- Example reviews
- Positive/Negative prediction
- Real-time inference

Run locally:

```bash
streamlit run app.py
```

## Future Improvements

- LSTM implementation
- GRU implementation
- Bidirectional RNN
- Attention Mechanism
- Better text preprocessing
- Improved model accuracy

---

## Learning Outcomes

- Natural Language Processing
- Deep Learning
- Recurrent Neural Networks
- TensorFlow/Keras
- Streamlit Deployment
- Model Evaluation

## Conclusion

In this project, a Sentiment Analysis System was successfully developed using a Simple Recurrent Neural Network (SimpleRNN) to classify IMDb movie reviews as positive or negative. The model was trained on the IMDb dataset, evaluated using standard classification metrics, and achieved satisfactory performance on unseen data.

Furthermore, the trained model was deployed as an interactive Streamlit web application, enabling users to perform real-time sentiment analysis through a modern and user-friendly interface. This project demonstrates the complete workflow of an end-to-end deep learning application, from data preprocessing and model development to evaluation and deployment.
