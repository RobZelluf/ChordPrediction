import pandas as pd
from sklearn.model_selection import train_test_split


def get_data():
    data_filename = "chord_data.csv"
    df = pd.read_csv(data_filename, header=None)

    """ Get target values """
    df_y = df.iloc[:, 0:2]
    y = [int(x[0]) * 2 + int(x[1]) for x in df_y.values]

    """ Get input values """
    df_x = df.iloc[:, 2:]

    """ Normalize input values """
    X = df_x.values
    X = (X - X.min()) / (X.max() - X.min())

    """ Split data into training and testing sets """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test
