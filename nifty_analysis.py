import sys
import matplotlib.pyplot as plt
from scripts.data_preprocessing import load_and_preprocess_data
from scripts.clustering import apply_kmeans_clustering
from scripts.pca_analysis import apply_pca
from scripts.anomaly_detection import detect_anomalies
from scripts.optimal_k_analysis import run_optimal_k_analysis

# Load the dataset
file_path = "data/nifty_dataset.csv"
df = load_and_preprocess_data(file_path)

# Function to display menu
def show_menu():
    print("\n NIFTY 50 Analysis Menu")
    print("1 - K-Means Clustering")
    print("2 - PCA (Principal Component Analysis)")
    print("3 - Anomaly Detection (Isolation Forest)")
    print("4 - Determine Optimal K for Clustering")
    print("5 - Run All Analyses")
    print("6 - Exit")
    return input("\nEnter your choice: ")

# Menu loop
while True:
    choice = show_menu()

    if choice == "1":
        print("\n Running Optimal K Analysis for Clustering...")
        run_optimal_k_analysis(df)
        plt.show(block=True)
        print("\n Running K-Means Clustering...")
        df, model = apply_kmeans_clustering(df, num_clusters=3)
        plt.show(block=True)

    elif choice == "2":
        print("\n Running PCA (Principal Component Analysis)...")
        pca_result, pca_model = apply_pca(df)
        plt.show(block=True)

    elif choice == "3":
        print("\n Running Anomaly Detection...")
        df, model = detect_anomalies(df)
        plt.show(block=True)

    elif choice == "4":
        print("\n🔍 Running All Analyses...")
        run_optimal_k_analysis(df)
        df, model = apply_kmeans_clustering(df, num_clusters=3)
        pca_result, pca_model = apply_pca(df)
        df, model = detect_anomalies(df)
        plt.show(block=True)

    elif choice == "5":
        print("Exiting... Have a great day!")
        sys.exit()

    else:
        print("Invalid choice! Please select a valid option.")
