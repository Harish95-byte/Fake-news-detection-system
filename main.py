from src.data_loader import load_data
from src.preprocessing import clean_text
from src.model_training import train_fake_news_model
from src.prediction import predict_news
from src.visualization import plot_distribution

def main():

    # Load dataset
    df = load_data()

    print(
        df.head()
    )

    # Clean text
    df['text'] = df['text'].apply(
        clean_text
    )

    # Train model
    model, vectorizer = train_fake_news_model(
        df
    )

    # Visualization
    plot_distribution(df)

    # User input
    news = input(
        "\nEnter News Text:\n"
    )

    news = clean_text(news)

    prediction = predict_news(
        model,
        vectorizer,
        news
    )

    if prediction == 0:

        print(
            "\nPrediction: Fake News"
        )

    else:

        print(
            "\nPrediction: Real News"
        )

if __name__ == "__main__":

    main()