import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load(path, test_size=0.2, random_state=1):
    data = pd.read_csv(path, sep=';', header=None)
    X, y = data.to_numpy()[:,:-1], data.to_numpy()[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test