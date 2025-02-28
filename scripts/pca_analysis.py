import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scripts.data_preprocessing import load_and_preprocess_data

def apply_pca(df, n_components=2):
    """ Apply PCA for dimensionality reduction """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(df)

    # Visualizing PCA results
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], cmap='coolwarm', alpha=0.6)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA: Stock Market Trends")
    plt.colorbar(label="Trend")

    return pca_result, pca  # Return model without auto-showing graph

if __name__ == "__main__":
    df = load_and_preprocess_data("..NiftyFiftyAnalysis/data/nifty_dataset.csv")
    pca_result, pca_model = apply_pca(df)
    plt.show(block=True)
