
import ssl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from ucimlrepo import fetch_ucirepo


ssl._create_default_https_context = ssl._create_unverified_context

def get_data(dataset_id, test_size=0.2, random_state=42):

    dataset = fetch_ucirepo(id=dataset_id)

    X = dataset.data.features
    y = dataset.data.targets

    lb = LabelBinarizer()
    y = lb.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, y_train, X_test, y_test

# letter_recognition = fetch_ucirepo(id=59)
# print(letter_recognition.metadata)
# print(letter_recognition.variables)


