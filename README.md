# AI Fake News Detection System

## Overview

The AI Fake News Detection System is an NLP-based Machine Learning project designed to classify news articles as Fake or Real using Natural Language Processing and Machine Learning techniques.

The system analyzes textual news content, converts it into numerical vectors using TF-IDF Vectorization, and predicts whether the news article is fake or genuine using a Logistic Regression classification model.

This project helps in understanding AI-based misinformation detection systems used in media monitoring and digital journalism platforms.

---

# Features

- Fake vs Real News Classification
- NLP-based text preprocessing
- TF-IDF Vectorization
- Logistic Regression Machine Learning Model
- Real-time News Prediction
- Fake News Detection Dashboard
- Dataset Visualization
- Interactive Streamlit Interface

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLP (Natural Language Processing)
- TF-IDF Vectorization
- Logistic Regression
- Matplotlib
- Streamlit

---

# Dataset

Dataset Used:
Fake and Real News Dataset (Kaggle)

Dataset Link:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Files Used:
- Fake.csv
- True.csv

Note:
The dataset is not uploaded to GitHub because of GitHub file size limitations.

After downloading, place the dataset inside:

Data/

---

# Project Structure

fake-news-detection-system/

├── Data/

├── models/

├── output/

├── src/

│   ├── data_loader.py

│   ├── preprocessing.py

│   ├── model_training.py

│   ├── prediction.py

│   ├── visualization.py

├── app.py

├── main.py

├── requirements.txt

├── README.md

├── .gitignore

---

# Installation

## Clone Repository

git clone <repository_link>

## Create Virtual Environment

python -m venv venv

## Activate Virtual Environment

Windows:

.\venv\Scripts\activate

## Install Dependencies

pip install -r requirements.txt

---

# Run Project

## Run Main Project

python main.py

## Run Streamlit Dashboard

streamlit run app.py

---

# Workflow

1. Load fake and real news datasets
2. Combine datasets and assign labels
3. Clean and preprocess news text
4. Convert text into TF-IDF vectors
5. Train Logistic Regression model
6. Predict fake or real news
7. Generate visualization analytics
8. Display results in Streamlit dashboard

---

# Output

The system generates:

- Fake or Real News Prediction
- Dataset Distribution Visualization
- Machine Learning Accuracy
- Interactive AI Dashboard

---

# Model Performance

Model Used:
- Logistic Regression

Feature Extraction:
- TF-IDF Vectorization

Achieved Accuracy:
- 98.44%

---

# Applications

- Fake News Detection
- Social Media Monitoring
- Media Verification Systems
- AI Journalism Analytics
- Misinformation Detection Platforms

---

# Future Enhancements

- Deep Learning-based News Classification
- BERT-based NLP Models
- Real-time News API Integration
- News Credibility Scoring
- Multi-language Fake News Detection

---

# Author

Harish Yuvaraj

AI/ML Internship Project
