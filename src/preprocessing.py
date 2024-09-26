import pandas as pd

def clean_data(data):
    """Remove ou preenche valores nulos nos dados."""
    print("Limpando os dados...")
    data_clean = data.fillna(data.mean())  # Preencher nulos com a média
    return data_clean

def remove_outliers(data):
    """Remove outliers dos dados usando o Z-score."""
    from scipy import stats
    print("Removendo outliers...")
    data_no_outliers = data[(abs(stats.zscore(data)) < 3).all(axis=1)]
    return data_no_outliers

if __name__ == "__main__":
    df = pd.read_csv('../data/dados.csv')
    df_clean = clean_data(df)
    df_no_outliers = remove_outliers(df_clean)
