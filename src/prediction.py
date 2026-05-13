def predict_news(
    model,
    vectorizer,
    news_text
):

    transformed_text = vectorizer.transform(
        [news_text]
    )

    prediction = model.predict(
        transformed_text
    )[0]

    return prediction