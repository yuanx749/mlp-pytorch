import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def load(path, test_size=0.2, random_state=1, cluster=None):
    data = pd.read_csv(path, sep=';', header=None)
    X, y = data.to_numpy()[:,:-1], data.to_numpy()[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    if cluster == 'kmeans':
        score = 0
        n_clusters_best = 2
        for n_clusters in range(2, 8):
            cluster_labels = KMeans(n_clusters=n_clusters, random_state=random_state).fit_predict(X_train)
            silhouette_avg = silhouette_score(X_train, cluster_labels)
            if silhouette_avg > score:
                score = silhouette_avg
                n_clusters_best = n_clusters
        kmeans = KMeans(n_clusters=n_clusters_best, random_state=random_state)
        labels_train = kmeans.fit_predict(X_train)
        labels_test = kmeans.predict(X_test)
        enc = OneHotEncoder()
        X_train_cluster = enc.fit_transform(labels_train[:,np.newaxis]).toarray()
        X_test_cluster = enc.transform(labels_test[:,np.newaxis]).toarray()
        X_train = np.hstack((X_train, X_train_cluster))
        X_test = np.hstack((X_test, X_test_cluster))
    return X_train, X_test, y_train, y_test