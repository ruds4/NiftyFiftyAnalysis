import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scripts.data_preprocessing import load_and_preprocess_data

def apply_kmeans_clustering(df, num_clusters=3):
    """ Apply K-Means clustering on the dataset """
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df)

    # Visualizing clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis', alpha=0.6)
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.title("Stock Clusters Based on Price Movements")
    plt.colorbar(label="Cluster")
    
    return df, kmeans  # Return model without auto-showing graph

if __name__ == "__main__":
    df = load_and_preprocess_data("..NiftyFiftyAnalysis/data/nifty_dataset.csv")
    df, model = apply_kmeans_clustering(df, num_clusters=3)
    plt.show(block=True)
