import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

#import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor
import optuna
from optuna import Trial
from functools import partial

from cell_TP_pred_model.config.core import config
from cell_TP_pred_model.processing.data_manager import load_dataset, replace_outliers_with_mean, create_and_save_model

#Split the data
def train_test_split_data(series, label):
    #Random splitting will not work for time series data. So below splitting is used
    tss = TimeSeriesSplit(n_splits = 3)

    # for train_index, test_index in tss.split(X):
    for train_index, test_index in tss.split(series):
        X_train, X_test = series[train_index, :], series[test_index,:]
        y_train, y_test = label[train_index], label[test_index]

    return X_train, y_train, X_test, y_test

# Define the Optuna optimization objective
def objective(trial: Trial, X_train, y_train, X_test, y_test):
    # Suggest values for hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 
                                        config.model_config_.min_learning_rate, 
                                        config.model_config_.max_learning_rate, step=0.01)
    max_depth = trial.suggest_int('max_depth',
                                  config.model_config_.min_max_depth, 
                                  config.model_config_.max_max_depth)
    n_estimators = trial.suggest_int('n_estimators', 
                                     config.model_config_.min_n_estimators, 
                                     config.model_config_.max_n_estimators, step=100)
    gamma = trial.suggest_float('gamma', 0, 1)
    min_child_weight = trial.suggest_int('min_child_weight', 
                                         config.model_config_.min_min_child_weight, 
                                         config.model_config_.max_min_child_weight)

    # Initialize the XGBoost model with the suggested hyperparameters
    model = XGBRegressor(
        objective='reg:squarederror',
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=n_estimators,
        gamma=gamma,
        min_child_weight=min_child_weight,
        random_state=config.model_config_.random_state,
    )

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the validation data
    y_pred = model.predict(X_test)

    # Calculate the RMSE (Root Mean Squared Error)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return rmse

def run_optuna(X_train, y_train, X_test, y_test):
    # # Create an Optuna study to minimize the objective (RMSE)
    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=50)  # Run 50 trials

    # Create a partial function that includes the additional arguments (X_train, y_train, X_test, y_test)
    objective_with_data = partial(objective, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    
    # Create an Optuna study to minimize the objective (RMSE)
    study = optuna.create_study(direction='minimize')
    
    # Run 50 trials with the updated objective function that includes the data
    study.optimize(objective_with_data, n_trials=50)
    
    # Print the best hyperparameters found by Optuna
    print("Best hyperparameters: ", study.best_params)
    return study

input_data_series_count = 5 #10  # Hyperparameter: number of previous samples to use for prediction

# Create input-output pairs for XGBoost (using the last N time steps to predict the next one)
def prepare_features_labels_for_xgboost(df_cleaned_zscore):
    features = df_cleaned_zscore.to_numpy()

    series = []
    label = []
    # for i in range(len(features) - input_data_series_count):
    #     series.append(features[i:i+input_data_series_count, 0])  # Input: Previous N samples
    #     label.append(features[i+input_data_series_count, 0])  # Label: Next sample

    for i in range(len(features) - input_data_series_count):
        # Get the previous N time steps (input data)
        input_data = features[i:i + input_data_series_count]
        mean_value = np.mean(input_data)

        # Append the input data along with the calculated mean as an additional feature
        series.append(np.append(input_data, mean_value))  # Input: Previous N samples + mean

        label.append(features[i+input_data_series_count, 0])  # Label: Next sample

    # if (DEBUG == True):
    #     print(series[0:4])
    #     print(label[:4])
    #     df_cleaned_zscore.head(12)

    # Convert lists to numpy arrays
    series = np.array(series)
    label = np.array(label)
    
    return series, label

def run_training() -> None:

    """
    Train the model.
    """

    # # read training data
    # data = load_dataset(file_name = config.app_config_.training_data_file)
    
    # # divide train and test
    # X_train, X_test, y_train, y_test = train_test_split(
        
    #     data[config.model_config_.features],     # predictors
    #     data[config.model_config_.target],       # target
    #     test_size = config.model_config_.test_size,
    #     random_state=config.model_config_.random_state,   # set the random seed here for reproducibility
    # )

    # # Pipeline fitting
    # bikeshare_pipe.fit(X_train, y_train)
    # y_pred = bikeshare_pipe.predict(X_test)

    # # Calculate the score/error
    # print("R2 score:", round(r2_score(y_test, y_pred), 2))
    # print("Mean squared error:", mean_squared_error(y_test, y_pred))

    # # persist trained model
    # save_pipeline(pipeline_to_persist = bikeshare_pipe)

    file_name = './attach_processed_Telcel_Daily_Performance_Report_cell_410001L001_data_modif.csv'
    df = load_dataset(file_name = config.app_config_.training_data_file)

    # if (DEBUG == True):
    #     print(df.head())

    #for col in df_cleaned.columns:
    df_cleaned_zscore = replace_outliers_with_mean(df, 'S1.ConnEstSucc', method='Z-Score')

    series, label = prepare_features_labels_for_xgboost(df_cleaned_zscore)

    X_train, y_train, X_test, y_test = train_test_split_data(series, label)
    study = run_optuna(X_train, y_train, X_test, y_test)
    create_and_save_model(study, X_train, y_train)

if __name__ == "__main__":
    run_training()