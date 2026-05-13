import matplotlib.pyplot as plt

def plot_distribution(df):

    counts = df['label'].value_counts()

    labels = [
        'Fake',
        'Real'
    ]

    plt.figure(
        figsize=(6,6)
    )

    plt.pie(
        counts,
        labels=labels,
        autopct='%1.1f%%'
    )

    plt.title(
        "Fake vs Real News Distribution"
    )

    plt.show()