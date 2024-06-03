import pandas as pd
from preprocessing import processingData
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import time
from plotting import plot_bar

df_por = processingData()
print(df_por.head())
print(df_por.info())

# dzielimy na X_data, Y_Data
X_data = df_por.drop(columns=['G3'])
Y_data = df_por['G3'] 

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
print("Rozmiar danych treningowych:", X_train.shape)
print("Rozmiar danych testowych:", X_test.shape)
                 
for column in X_train.columns:
    if X_train[column].dtype == 'object':
        X_train[column] = X_train[column].astype(float)
    if X_test[column].dtype == 'object':
        X_test[column] = X_test[column].astype(float)

xgb_model = XGBRegressor()
start = time.time()
xgb_model.fit(X_train, Y_train)
Y_pred = xgb_model.predict(X_test)
end = time.time()
print(f"XGBoost fitting and predicting: {end-start}")

mse = mean_squared_error(Y_test, Y_pred)
print("Błąd średniokwadratowy (MSE):", mse)




def assign_grade(value):
    if value <= 6:
        return 'bardzo niska'
    if value <= 10:
        return 'niska'
    elif value <= 14:
        return 'średnia'
    elif value <=17:
        return 'wysoka'
    else:
        return 'bardzo wysoka'

Y_pred_class = [assign_grade(int(value)) for value in Y_pred]
Y_test_class = [assign_grade(int(value)) for value in Y_test]


accuracy = sum(1 for p, t in zip(Y_pred_class, Y_test_class) if p == t) / len(Y_test_class)
print("Dokładność klasyfikacji:", accuracy)

plot_bar(Y_test_class, Y_pred_class)



