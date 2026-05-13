import pandas as pd

def load_data():

    fake_df = pd.read_csv(
        'Data/Fake.csv'
    )

    true_df = pd.read_csv(
        'Data/True.csv'
    )

    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat(
        [fake_df, true_df],
        ignore_index=True
    )

    return df