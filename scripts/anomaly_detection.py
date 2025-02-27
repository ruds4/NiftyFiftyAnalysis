import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from scripts.data_preprocessing import load_and_preprocess_data

def detect_anomalies(df, contamination=0.05):
    """ Detect anomalies using Isolation Forest """
    model = IsolationForest(contamination=contamination, random_state=42)
    df['Anomaly'] = model.fit_predict(df)

    # Visualizing anomalies
    plt.figure(figsize=(10, 6))
    plt.scatter(df.index, df.iloc[:, 0], c=df['Anomaly'], cmap='coolwarm', alpha=0.6)
    plt.xlabel("Date")
    plt.ylabel(df.columns[0])
    plt.title("Anomaly Detection in Stock Prices")
    plt.colorbar(label="Anomaly (1: Normal, -1: Anomaly)")

    return df, model  # Return model without auto-showing graph

if __name__ == "__main__":
    df = load_and_preprocess_data("../data/nifty_dataset.csv")
    df, model = detect_anomalies(df)
    plt.show(block=True)
