import pandas as pd
from preprocessing import processingData
from modeling import modeling_data_xgboost, modeling_data_randomforestregressor, modeling_data_gbr
from plotting import plot_bar
from eda import correlation_matrix
#XGBoost

# portugalski 
file_path_por = './data/student-por.csv'
df_por_raw = pd.read_csv(file_path_por)
df_por = processingData(file_path_por)
Y_pred_class_por, Y_test_class_por  = modeling_data_xgboost(file_path_por)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")

# matematyka
file_path_mat = './data/student-mat.csv'
df_mat = processingData(file_path_mat)
Y_pred_class_mat, Y_test_class_mat  = modeling_data_xgboost(file_path_mat)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "matematyki")

#RandomTreeForestRegressor

# portugalski 
Y_pred_class_por, Y_test_class_por  = modeling_data_randomforestregressor(file_path_por)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")

# matematyka
Y_pred_class_por, Y_test_class_por  = modeling_data_randomforestregressor(file_path_mat)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")

# GradientBoostingRegressor

# portugalski
Y_pred_class_mat, Y_test_class_mat  = modeling_data_gbr(file_path_mat)
plot_bar(Y_test_class_mat, Y_pred_class_mat, 'G3', "matematyki")
# matematyka
Y_pred_class_por, Y_test_class_por  = modeling_data_gbr(file_path_por)
plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")

# EDA
correlation_matrix(df_por)
# najmniej skorelowane z G3 (ponizej modulo 0.10) - abscences, freetime, goout, famrel, nursery, fjob, schoolsup, paid, activities, pstatus, famsize
# najbardziej skorelowane z G3 top - failures, higher, school, medu, fedu
correlation_matrix(df_mat)

#Future Processing
# ze wzgledu ze wszystkie zmienne poza targetem są kategoryczne, nie występują outliery. Dodatkowo nulle w tym zbiorze nie wystepuja. 












