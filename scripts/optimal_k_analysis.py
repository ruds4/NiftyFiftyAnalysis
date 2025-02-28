import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scripts.data_preprocessing import load_and_preprocess_data

# Load the preprocessed dataset (cleaning is handled in the function)
file_path = "NiftyFiftyAnalysis/data/nifty_dataset.csv"
df = load_and_preprocess_data(file_path)

def run_optimal_k_analysis():
    # Elbow Method Analysis
    # Define the range of k values for the elbow method
    k_range_elbow = range(1, 11)
    sse = []

    for k in k_range_elbow:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    # Silhouette Score Analysis for k = 2, 3, 4
    k_values_sil = [2, 3, 4]
    sil_scores = []

    for k in k_values_sil:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(df)
        score = silhouette_score(df, labels)
        sil_scores.append(score)
        print(f"Silhouette score for k = {k}: {score:.4f}")

    # Plotting the results

    plt.figure(figsize=(12, 5))

    # Plot Elbow Method
    plt.subplot(1, 2, 1)
    plt.plot(list(k_range_elbow), sse, marker='o', linestyle='--')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Sum of Squared Errors (SSE)")
    plt.title("Elbow Method for Optimal k")

    # Plot Silhouette Score Analysis
    plt.subplot(1, 2, 2)
    plt.plot(k_values_sil, sil_scores, marker='o', linestyle='--', color='red')
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score Analysis (k = 2, 3, 4)")

    plt.tight_layout()
    plt.show()
