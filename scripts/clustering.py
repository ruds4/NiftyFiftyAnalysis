import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scripts.data_preprocessing import load_and_preprocess_data

# Load and preprocess the dataset
file_path = "NiftyFiftyAnalysis/data/nifty_dataset.csv"
df = load_and_preprocess_data(file_path)

def clusters_with_pca():
    # Apply K-Means clustering with k = 2
    k = 2
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(df)

    # Reduce dimensions for visualization using PCA (to 2 components)
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=labels, palette='viridis', s=50, alpha=0.8)
    plt.title("Clusters Visualized with PCA (k = 2)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()
