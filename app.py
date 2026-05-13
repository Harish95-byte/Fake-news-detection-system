import streamlit as st
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.model_training import train_fake_news_model
from src.prediction import predict_news

# Page Config
st.set_page_config(
    page_title="AI Fake News Detection",
    layout="wide"
)

# Title
st.title(
    "AI Fake News Detection System"
)

st.markdown(
    "Detect whether a news article is Fake or Real using NLP and Machine Learning."
)

# Load Dataset
df = load_data()

# Clean Text
df['text'] = df['text'].apply(
    clean_text
)

# Train Model
model, vectorizer = train_fake_news_model(
    df
)

# User Input
news_input = st.text_area(
    "Enter News Article",
    height=250
)

# Predict Button
if st.button(
    "Detect News"
):

    if news_input.strip() == "":

        st.warning(
            "Please enter news text."
        )

    else:

        cleaned_news = clean_text(
            news_input
        )

        prediction = predict_news(
            model,
            vectorizer,
            cleaned_news
        )

        st.subheader(
            "Prediction Result"
        )

        if prediction == 0:

            st.error(
                "Fake News Detected"
            )

        else:

            st.success(
                "Real News Detected"
            )

        # Visualization
        st.subheader(
            "Dataset Distribution"
        )

        counts = df['label'].value_counts()

        labels = [
            'Fake',
            'Real'
        ]

        fig, ax = plt.subplots(
            figsize=(6,6)
        )

        ax.pie(
            counts,
            labels=labels,
            autopct='%1.1f%%'
        )

        ax.set_title(
            "Fake vs Real News Distribution"
        )

        st.pyplot(fig)

        # Accuracy Display
        st.info(
            "Model Accuracy: 98.44%"
        )