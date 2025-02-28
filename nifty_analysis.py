import sys
import matplotlib.pyplot as plt
from scripts.data_preprocessing import load_and_preprocess_data
from scripts.clustering import clusters_with_pca
from scripts.anomaly_detection import detect_anomalies
from scripts.optimal_k_analysis import run_optimal_k_analysis

# Load the dataset
file_path = "NiftyFiftyAnalysis/data/nifty_dataset.csv"
df = load_and_preprocess_data(file_path)

def main_menu():
    while True:
        print("\n=== NIFTY Analysis Menu ===")
        print("1 - Run Clustering Analysis (Elbow, Silhouette, and PCA Visualization)")
        print("2 - Run Anomaly Detection")
        print("4 - Exit")
        choice = input("Enter your choice: ")
        
        if choice == "1":
            print("Running optimal k analysis...")
            run_optimal_k_analysis()
            print("Performing PCA for cluster visualization...")
            clusters_with_pca()
        elif choice == "2":
            detect_anomalies()
        elif choice == "4":
            print("Exiting... Have a great day!")
            sys.exit()
        else:
            print("Invalid choice! Please select a valid option.")

if __name__ == "__main__":
    main_menu()
