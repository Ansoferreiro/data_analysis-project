import pandas as pd

def load_data(filepath):
    """Carrega os dados de um arquivo CSV."""
    try:
        data = pd.read_csv(filepath)
        print("Dados carregados com sucesso!")
        return data
    except FileNotFoundError:
        print("Arquivo não encontrado.")
        return None

def show_data_info(data):
    """Mostra informações gerais dos dados."""
    print(data.info())
    print(data.describe())

if __name__ == "__main__":
    df = load_data('../data/dados.csv')
    if df is not None:
        show_data_info(df)
