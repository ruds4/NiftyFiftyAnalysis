import pandas as pd

def load_and_preprocess_data(file_path):
    """ Load the dataset and perform preprocessing """
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime if exists
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Keep only numeric columns
    df_numeric = df.select_dtypes(include=['number'])

    # Handle missing values
    df_numeric = df_numeric.ffill()

    return df_numeric

if __name__ == "__main__":
    df = load_and_preprocess_data("../data/nifty_dataset.csv")
    print(df.head())  # Preview cleaned data
