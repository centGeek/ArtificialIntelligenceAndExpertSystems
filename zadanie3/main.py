from preprocessing import processingData
from modeling import modeling_data_xgboost, modeling_data_randomforestregressor, modeling_data_gbr
from plotting import plot_bar, plot_errors
from exploring import correlation_matrix, futureProcessing
file_path_por = './data/student-por.csv'
def testing_models(do_future_processing: bool):
    df_por = processingData(file_path_por)
    if do_future_processing:
        df_por = futureProcessing(df_por)
    # XGBoost

    # portugalski
    Y_pred_class_por, Y_test_class_por, mse_xgb_por, mae_xgb_por, r2_xgb_por  = modeling_data_xgboost(file_path_por)
    plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")

    file_path_mat = './data/student-mat.csv'
    df_mat = processingData(file_path_mat)
    if do_future_processing:
        df_mat = futureProcessing(df_mat)
    
    # matematyka
    Y_pred_class_mat, Y_test_class_mat, mse_xgb_mat, mae_xgb_mat, r2_xgb_mat  = modeling_data_xgboost(file_path_mat)
    plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "matematyki")

    #RandomTreeForestRegressor

    # portugalski 
    Y_pred_class_por, Y_test_class_por, mse_random_forest_por, mae_random_forest_por, r2_random_forest_por = modeling_data_randomforestregressor(file_path_por)
    plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")

    # matematyka
    Y_pred_class_mat, Y_test_class_mat, mse_random_forest_mat, mae_random_forest_mat, r2_random_forest_mat = modeling_data_randomforestregressor(file_path_mat)
    plot_bar(Y_test_class_mat, Y_pred_class_mat, 'G3', "matematyki")

    # GradientBoostingRegressor

    # portugalski
    Y_pred_class_por, Y_test_class_por, mse_gbr_por,mae_gbr_por, r2_gbr_por = modeling_data_gbr(file_path_por)
    plot_bar(Y_test_class_por, Y_pred_class_por, 'G3', "portugalskiego")
    
    # matematyka
    Y_pred_class_mat, Y_test_class_mat, mse_gbr_mat,mae_gbr_mat, r2_gbr_mat= modeling_data_gbr(file_path_mat)
    plot_bar(Y_test_class_mat, Y_pred_class_mat, 'G3', "matematyki")

    plot_errors(mse_xgb_por, mae_xgb_por, mse_xgb_mat, mae_xgb_mat, mse_random_forest_por, mae_random_forest_por, mse_random_forest_mat, mae_random_forest_mat, mse_gbr_por,mae_gbr_por, mse_gbr_mat,mae_gbr_mat, r2_xgb_por, r2_xgb_mat,r2_random_forest_por, r2_random_forest_mat, r2_gbr_por, r2_gbr_mat)
    # EDA
    correlation_matrix(df_por)
    # najmniej skorelowane z G3 (ponizej modulo 0.10) - abscences, freetime, goout, famrel, nursery, fjob, schoolsup, paid, activities, pstatus, famsize
    # najbardziej skorelowane z G3 top - failures, higher, school, medu, fedu
    correlation_matrix(df_mat)

#Future Processing
# ze wzgledu ze wszystkie zmienne poza targetem są kategoryczne, nie występują outliery. Dodatkowo nulle w tym zbiorze nie wystepuja. 
testing_models(do_future_processing=False)

testing_models(do_future_processing=True)














