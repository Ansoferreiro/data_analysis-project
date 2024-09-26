import matplotlib.pyplot as plt

def plot_real_vs_pred(y_test, y_pred, output_path):
    """Plota a comparação entre valores reais e previstos."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel('Valores Reais')
    plt.ylabel('Previsões')
    plt.title('Previsões vs Valores Reais')
    plt.savefig(output_path)

if __name__ == "__main__":
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    # Exemplo de uso
    df = pd.read_csv('../data/dados.csv')
    X = df[['coluna1', 'coluna2']]
    y = df['alvo']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    plot_real_vs_pred(y_test, y_pred, '../reports/previsoes_vs_reais.png')
