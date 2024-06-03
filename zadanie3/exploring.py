import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()

def futureProcessing(data : pd.DataFrame):
    # z powodu najmniejszej korelacji tych zmiennych, sprobujmy usunac je z naszego datasetu
    columns_to_drop = ['activities', 'pstatus', 'famsize']
    df = data.drop(columns=columns_to_drop)
    return df
