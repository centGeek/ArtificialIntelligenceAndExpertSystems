import pandas as pd
from preprocessing import processingData
from modeling import modeling_data_xgboost, modeling_data_randomforestregressor
from plotting import plot_bar
#XGBoost

# portugalski 
file_path_por = './data/student-por.csv'
df_por = processingData(file_path_por)
Y_pred_class_por, Y_test_class_por  = modeling_data_xgboost(file_path_por)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")
# matematyka
file_path_mat = './data/student-mat.csv'
df_mat = processingData(file_path_mat)
Y_pred_class_mat, Y_test_class_mat  = modeling_data_xgboost(file_path_mat)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "matematyki")

#RandomTreeForstRegressor

# portugalski 
Y_pred_class_por, Y_test_class_por  = modeling_data_randomforestregressor(file_path_por)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")
# matematyka
Y_pred_class_mat, Y_test_class_mat  = modeling_data_randomforestregressor(file_path_mat)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "matematyki")



# eda







