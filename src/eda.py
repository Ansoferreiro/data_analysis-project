import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(data, column, output_path):
    """Plota um histograma de uma coluna."""
    plt.figure(figsize=(8, 6))
    data[column].hist(bins=30)
    plt.title(f'Histograma de {column}')
    plt.xlabel(column)
    plt.ylabel('Frequência')
    plt.savefig(output_path)

def plot_correlation_matrix(data, output_path):
    """Plota uma matriz de correlação."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Matriz de Correlação')
    plt.savefig(output_path)

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv('../data/dados.csv')
    plot_histogram(df, 'coluna_interessante', '../reports/histograma.png')
    plot_correlation_matrix(df, '../reports/correlacao.png')
