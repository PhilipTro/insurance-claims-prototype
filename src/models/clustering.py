from ast import Index
from typing import Any
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.model_selection import GridSearchCV

ELBOW_INERTIA = 2

def load_data(filepath):
    return pd.read_csv(filepath)

def kmeans_cluster(X, random_state=42):
    sil = -1
    for n_clusters in range(2,11):
        km = KMeans(n_clusters=n_clusters, init='k-means++', n_init=n_clusters, random_state=random_state)
        model = km.fit(X)
        labels = model.labels_
        test_sil = silhouette_score(X, labels=labels, metric='euclidean')
        if sil < test_sil:
            best_n = n_clusters
            sil = test_sil
    best_model = KMeans(n_clusters=best_n, init='k-means++', n_init=best_n, random_state=random_state).fit(X)
    labels = best_model.labels_
    return labels, best_model, sil

def kmeans_add_labels_and_cluster_centers(df_clean:pd.DataFrame, labels:np.ndarray, features) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Function for adding labels to original dataset, and for producing a dataframe of the final cluster centroids

    Args:
        df_clean (pd.DataFrame): Cleaned dataframe (not imputed or scaled)
        labels (np.ndarray): Labels resulting from clustering algorithm
        features (list[str]): Feature names

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: tuple of labeled dataset and centroids dataframe
    """
    df_labeled = df_clean.copy()
    df_labeled["Cluster kmeans"] = labels
    centroids = df_labeled.groupby("Cluster kmeans")[features].mean()
    df_labeled.to_csv("./data/clustering/KMEANS_df_with_labels.csv")
    return df_labeled, centroids

def plot_clusters_2d(X_pca:np.ndarray, labels:np.ndarray, title:str = "Cluster plot", model:str='KMeans') -> None:
    """function for plotting the clusters in two dimensions, using PCA component 1, and PCA component 2

    Args:
        X_pca (np.ndarray): PCA data matrix
        labels (np.ndarray): labels from clustering algorithm
        title (str, optional): Title of the plot. Defaults to "Cluster plot".
        model (str, optional): Algorithm used for clustering. Defaults to 'KMeans'.
    """
    save_path = f"graphs/clustering/{model}/{title}.png"
    plt.figure(figsize=(10,8))
    un_labels = np.unique(labels)

    for label in un_labels:
        plt.scatter(
            X_pca[labels==label, 0],
            X_pca[labels==label, 1],
            label = f"Cluster {label}",
            alpha = 0.6
        )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches = 'tight')
    plt.show()


def elbow_inertia(X:np.ndarray | pd.DataFrame, k_min:int = 1, k_max:int = 10, random_state:int = 42) -> list[float]:
    """Function for collecting interitias from different amounts of clusters when implementing Kmeans

    Args:
        X (np.ndarray): clustering data
        k_min (int, optional): lower boundary of n_clusters to test. Defaults to 1.
        k_max (int, optional): upper boundary of n_clusters to test. Defaults to 10.
        random_state (int, optional): Random state for reproducability. Defaults to 42.

    Returns:
        list[float]: List of inertias
    """
    inertias = [KMeans(n_clusters=k, init='k-means++', n_init=k, random_state=random_state).fit(X).inertia_ for k in range(k_min, k_max+1)]
    assert len(inertias) == (k_max - k_min +1)
    return inertias 

def dbscan_clusters(X:np.ndarray|pd.DataFrame) -> tuple[np.ndarray, DBSCAN, float, int]:
    """Function for implementing DBSCAN cluster algorithm

    Args:
        X (np.ndarray): Clustering data

    Returns:
        tuple[np.ndarray, DBSCAN, float, int]: Tuple of labels, DBSCAN-model, best ep and best min for the model.
    """
    eps = [i for i in np.arange(0.2, 10, step=0.2)]
    min_samples = np.arange(5, 21, 1)
    sil = -1
    for min_sample in min_samples:
        for ep in eps:
            model = DBSCAN(eps=ep, min_samples=min_sample).fit(X)
            labels = model.labels_
            unique_labels = set(labels)
            if len(unique_labels - {-1}) > 1 and len(unique_labels - {-1}) < len(X):
                test_sil = silhouette_score(X=X, labels=labels, metric='euclidean')
                if sil < test_sil:
                    best_ep = ep
                    best_min = min_sample
                    sil = test_sil
    best_model = DBSCAN(eps=best_ep, min_samples=best_min).fit(X)
    labels = best_model.labels_
    return labels, best_model, float(best_ep), int(best_min)

def add_dbscan_labels(df_clean:pd.DataFrame, labels:np.ndarray, features) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_labeled = df_clean.copy()
    df_labeled["Cluster dbscan"] = labels
    centroids = df_labeled.groupby("Cluster dbscan")[features].mean()
    df_labeled.to_csv("./data/clustering/DBSCAN_df_with_labels.csv")
    return df_labeled, centroids