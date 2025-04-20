import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import pytest
#from sklearn.model_selection import train_test_split

from cell_TP_pred_model.config.core import config
from cell_TP_pred_model.processing.data_manager import load_dataset, replace_outliers_with_mean
from cell_TP_pred_model.train_pipeline import prepare_features_labels_for_xgboost, train_test_split_data


@pytest.fixture
def sample_input_data():
    
    df = load_dataset(file_name = config.app_config_.training_data_file)

    #for col in df_cleaned.columns:
    df_cleaned_zscore = replace_outliers_with_mean(df, 'S1.ConnEstSucc', method='Z-Score')

    series, label = prepare_features_labels_for_xgboost(df_cleaned_zscore)

    X_train, y_train, X_test, y_test = train_test_split_data(series, label)

    return X_test, y_test    