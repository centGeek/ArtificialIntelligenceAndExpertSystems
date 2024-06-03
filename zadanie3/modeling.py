import pandas as pd
from preprocessing import processingData, assign_grade
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import time
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

def splitting_data(file_path):
    df = processingData(file_path)
    X_data = df.drop(columns=['G3'])
    Y_data = df['G3'] 
    X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
    print("Rozmiar danych treningowych:", X_train.shape)
    print("Rozmiar danych testowych:", X_test.shape)
                    
    for column in X_train.columns:
        if X_train[column].dtype == 'object':
            X_train[column] = X_train[column].astype(float)
        if X_test[column].dtype == 'object':
            X_test[column] = X_test[column].astype(float)
    return Y_train, Y_test, X_train, X_test

def modeling_data_xgboost(file_path):
    Y_train, Y_test, X_train, X_test= splitting_data(file_path)
    xgb_model = XGBRegressor()
    start = time.time()
    xgb_model.fit(X_train, Y_train)
    Y_pred = xgb_model.predict(X_test)
    end = time.time()
    print(f"Time XGBoost fitting and predicting: {end-start}")

    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error (MSE):", mse)

    mae = mean_absolute_error(Y_test, Y_pred)
    print("Mean Absolute Error (MAE):", mae)
    r2 = r2_score(Y_test, Y_pred)
    print("R-squared (R^2):", r2)

    Y_pred_class= [assign_grade(int(value)) for value in Y_pred]
    Y_test_class = [assign_grade(int(value)) for value in Y_test]


    accuracy = sum(1 for p, t in zip(Y_pred_class, Y_test_class) if p == t) / len(Y_test_class)
    print("Classification Accuracy:", accuracy)
    return Y_pred_class, Y_test_class, mse, mae, r2
def modeling_data_randomforestregressor(file_path, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features='sqrt'):
    Y_train, Y_test, X_train, X_test = splitting_data(file_path)
    rf_model = RandomForestRegressor(
        # n_estimators=n_estimators,
        # max_depth=max_depth,
        # min_samples_split=min_samples_split,
        # min_samples_leaf=min_samples_leaf,
        # max_features=max_features
    )
    
    start = time.time()
    rf_model.fit(X_train, Y_train)
    Y_pred = rf_model.predict(X_test)
    end = time.time()
    print(f"Time RandomForest fitting and predicting: {end - start}")
    
    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error (MSE):", mse)

    mae = mean_absolute_error(Y_test, Y_pred)
    print("Mean Absolute Error (MAE):", mae)

    r2 = r2_score(Y_test, Y_pred)
    print("R-squared (R^2):", r2)

    Y_pred_class = [assign_grade(int(value)) for value in Y_pred]
    Y_test_class = [assign_grade(int(value)) for value in Y_test]
    
    accuracy = sum(1 for p, t in zip(Y_pred_class, Y_test_class) if p == t) / len(Y_test_class)
    print("Classification Accuracy:", accuracy)
    
    return Y_pred_class, Y_test_class, mse, mae, r2
def modeling_data_svm(file_path):
    Y_train, Y_test, X_train, X_test = splitting_data(file_path)
    svm_model = SVR()
    
    start = time.time()
    svm_model.fit(X_train, Y_train)
    Y_pred = svm_model.predict(X_test)
    end = time.time()
    print(f"Time SVM fitting and predicting: {end - start}")
    
    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error (MSE):", mse)

    mae = mean_absolute_error(Y_test, Y_pred)
    print("Mean Absolute Error (MAE):", mae)

    r2 = r2_score(Y_test, Y_pred)
    print("R-squared (R^2):", r2)
    
    Y_pred_class = [assign_grade(int(value)) for value in Y_pred]
    Y_test_class = [assign_grade(int(value)) for value in Y_test]
    
    accuracy = sum(1 for p, t in zip(Y_pred_class, Y_test_class) if p == t) / len(Y_test_class)
    print("Classification Accuracy:", accuracy)
    
    return Y_pred_class, Y_test_class, mse, mae, r2
def modeling_data_gbr(file_path):
    Y_train, Y_test, X_train, X_test = splitting_data(file_path)    

    gbr_model = GradientBoostingRegressor()
    start = time.time()
    gbr_model.fit(X_train, Y_train)
    Y_pred = gbr_model.predict(X_test)
    end = time.time()
    print(f"Time GBR fitting and predicting: {end - start}")
    
    mse = mean_squared_error(Y_test, Y_pred)
    print("Mean Squared Error (MSE):", mse)

    mae = mean_absolute_error(Y_test, Y_pred)
    print("Mean Absolute Error (MAE):", mae)

    r2 = r2_score(Y_test, Y_pred)
    print("R-squared (R^2):", r2)
    
    Y_pred_class = [assign_grade(int(value)) for value in Y_pred]
    Y_test_class = [assign_grade(int(value)) for value in Y_test]
    
    accuracy = sum(1 for p, t in zip(Y_pred_class, Y_test_class) if p == t) / len(Y_test_class)
    print("Classification Accuracy:", accuracy)
    
    return Y_pred_class, Y_test_class, mse, mae, r2
