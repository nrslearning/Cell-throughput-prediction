import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import typing as t
from pathlib import Path

#import joblib
import pandas as pd
import numpy as np
#from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBRegressor

from cell_TP_pred_model import __version__ as _version
from cell_TP_pred_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config

def load_dataset(file_name):
    data_frame = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"), #file_name,
                            parse_dates=['Time'], dayfirst=True,
                            index_col="Time")
    data_frame['S1.ConnEstSucc'] = data_frame['S1.ConnEstSucc'].astype(float)

    # Drop unnecessary fields
    for field in config.model_config_.unused_fields:
        if field in data_frame.columns:
            data_frame.drop(labels = field, axis=1, inplace=True)   
    
    # #Drop id column
    # df.drop(columns=['id'], inplace=True) 

    return data_frame

#Replace outliers with mean values
def replace_outliers_with_mean(df, column, method='IQR'):
    # Calculate the mean of the column
    mean_value = df[column].mean()

    if method == 'IQR':
        # Calculate Q1, Q3, and IQR for IQR method
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with the mean value using .loc to avoid the warning
        df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = mean_value

    elif method == 'Z-Score':
        # Calculate Z-scores for Z-Score method
        z_scores = (df[column] - df[column].mean()) / df[column].std()

        # Replace outliers (Z-score > 3 or Z-score < -3) with the mean value
        df.loc[np.abs(z_scores) > 3, column] = mean_value

    else:
        raise ValueError("Invalid method. Choose either 'IQR' or 'Z-Score'.")

    return df

def create_and_save_model(study, X_train, y_train):
    # Train the final model with the best hyperparameters
    best_params = study.best_params
    final_model = XGBRegressor(
        objective='reg:squarederror',
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        n_estimators=best_params['n_estimators'],
        gamma=best_params['gamma'],
        min_child_weight=best_params['min_child_weight'],
        random_state=42
    )

    # Prepare versioned save file name
    save_file_name = f"{config.app_config_.model_save_file}{_version}.json"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])

    # Train the final model
    final_model.fit(X_train, y_train)

    final_model.save_model(save_path)
    print("Model saved successfully!")
    
    return

def load_model(*, file_name: str):
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = xgb.Booster()
    trained_model.load_model(file_path)

    return trained_model

def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one mapping between the package version and 
    the model version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()