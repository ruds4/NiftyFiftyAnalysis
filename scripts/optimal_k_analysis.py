import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scripts.data_preprocessing import load_and_preprocess_data

# Load dataset
file_path = "NiftyFiftyAnalysis/data/nifty_dataset.csv"
df = load_and_preprocess_data(file_path)

# Function to determine the optimal number of clusters
def run_optimal_k_analysis(df, k_range=(2, 10)):
    df_clean = df.dropna().select_dtypes(include=[np.number])
    
    sse = []  # Sum of Squared Errors for Elbow Method
    silhouette_scores = []  # Silhouette scores
    
    for k in range(k_range[0], k_range[1]):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df_clean)
        
        sse.append(kmeans.inertia_)  # Inertia for Elbow Method
        silhouette_scores.append(silhouette_score(df_clean, labels))  # Silhouette Score
    
    # Plot Elbow Method
    plt.figure(figsize=(10, 5))
    plt.plot(range(k_range[0], k_range[1]), sse, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.title("Elbow Method for Optimal k")
    plt.show()
    
    # Plot Silhouette Score
    plt.figure(figsize=(10, 5))
    plt.plot(range(k_range[0], k_range[1]), silhouette_scores, marker='o', linestyle='--', color='red')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Analysis for Optimal k")
    plt.show()

if __name__ == "__main__":
    run_optimal_k_analysis(df)
