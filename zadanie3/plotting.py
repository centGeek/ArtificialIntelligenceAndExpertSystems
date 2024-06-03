import matplotlib.pyplot as plt
import numpy as np


def count_values(data):
    unique_values, counts = np.unique(data, return_counts=True)
    return dict(zip(unique_values, counts)) #slownik - klucz (unikalne wartosci) - values (liczby wystapien)


def plot_bar(Y_test, Y_pred, grade, subject):
    actual_counts = count_values(Y_test)
    predicted_counts = count_values(Y_pred)
    all_labels = set(list(actual_counts.keys()) + list(predicted_counts.keys()))
    sorted_labels = sorted(all_labels, key=lambda x: actual_counts.get(x, 0))
    sorted_actual_counts = [actual_counts.get(label, 0) for label in sorted_labels]
    sorted_predicted_counts = [predicted_counts.get(label, 0) for label in sorted_labels]

    labels = list(sorted_labels)
    n = len(labels)
    bar_width = 0.4 

    plt.figure(figsize=(10, 5))

    
    plt.bar(np.arange(n), sorted_actual_counts, color='blue', width=bar_width, alpha=0.7, label='Rzeczywiste')

    plt.bar(np.arange(n) + bar_width, sorted_predicted_counts, color='orange', width=bar_width, alpha=0.7, label='Przewidywane')

    plt.xlabel(f'Wartość {grade}')
    plt.ylabel('Liczba wystąpień')
    plt.title(f'Liczba wystąpień rzeczywistych i przewidywanych wartości {grade}  dla toku ksztalcenia {subject}')
    plt.xticks(np.arange(n) + bar_width / 2, labels, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout() 
    plt.show()

def plot_errors(mse_xgb_por, mae_xgb_por, mse_xgb_mat, mae_xgb_mat, 
                mse_random_forest_por, mae_random_forest_por, mse_random_forest_mat, mae_random_forest_mat, 
                mse_gbr_por, mae_gbr_por, mse_gbr_mat, mae_gbr_mat,
                r2_xgb_por, r2_xgb_mat, r2_random_forest_por, r2_random_forest_mat, r2_gbr_por, r2_gbr_mat):
    
    models = ['XGBoost', 'Random Forest', 'Gradient Boosting']
    metrics = ['MSE', 'MAE', 'R^2']
    languages = ['Portuguese', 'Mathematics']
    
    colors = ['blue', 'green', 'red']
    
    mse_por = [mse_xgb_por, mse_random_forest_por, mse_gbr_por]
    mae_por = [mae_xgb_por, mae_random_forest_por, mae_gbr_por]
    r2_por = [r2_xgb_por, r2_random_forest_por, r2_gbr_por]
    
    mse_mat = [mse_xgb_mat, mse_random_forest_mat, mse_gbr_mat]
    mae_mat = [mae_xgb_mat, mae_random_forest_mat, mae_gbr_mat]
    r2_mat = [r2_xgb_mat, r2_random_forest_mat, r2_gbr_mat]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16)
    
    for i, language in enumerate(languages):
        for j, metric in enumerate(metrics):
            axes[i, j].bar(models, mse_por if metric == 'MSE' and language == 'Portuguese' else 
                                    (mae_por if metric == 'MAE' and language == 'Portuguese' else 
                                    (r2_por if metric == 'R^2' and language == 'Portuguese' else 
                                    (mse_mat if metric == 'MSE' and language == 'Mathematics' else 
                                    (mae_mat if metric == 'MAE' and language == 'Mathematics' else r2_mat)))),
                                    color=colors)
            axes[i, j].set_title(f'{language} - {metric}')
            axes[i, j].set_ylabel(metric)
        
    plt.tight_layout()
    plt.show()