from src.load_data import load_data, show_data_info
from src.preprocessing import clean_data, remove_outliers
from src.eda import plot_histogram, plot_correlation_matrix
from src.model import train_model, evaluate_model
from src.visualization import plot_real_vs_pred

def main():
    # Carregar dados
    data = load_data('data/dados.csv')
    if data is None:
        return
    
    # Mostrar informações dos dados
    show_data_info(data)
    
    # Limpar dados
    data_clean = clean_data(data)
    data_no_outliers = remove_outliers(data_clean)
    
    # Análise exploratória
   # Substitua 'coluna_interessante' por uma coluna existente
    plot_histogram(data_no_outliers, 'coluna_1', 'reports/histograma.png') 
    plot_correlation_matrix(data_no_outliers, 'reports/correlacao.png')
    
    # Preparar dados para o modelo
    X = data_no_outliers[['coluna_1', 'coluna_2']]
    y = data_no_outliers['alvo']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Treinar e avaliar o modelo
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, 'reports/model_performance.txt')
    
    # Visualizar os resultados
    y_pred = model.predict(X_test)
    plot_real_vs_pred(y_test, y_pred, 'reports/previsoes_vs_reais.png')

if __name__ == "__main__":
    main()
