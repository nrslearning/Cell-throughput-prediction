import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import Union
#import pandas as pd
import numpy as np
import xgboost as xgb

from cell_TP_pred_model import __version__ as _version
from cell_TP_pred_model.config.core import config
from cell_TP_pred_model.processing.data_manager import load_model
#from cell_TP_pred_model.processing.data_manager import pre_pipeline_preparation
from cell_TP_pred_model.processing.validation import validate_inputs

model_file_name = f"{config.app_config_.model_save_file}{_version}.json"
cell_TP_model = load_model(file_name = model_file_name)

def make_prediction(input_data: list[float]) -> dict:
    """Make a prediction given a list of 5 float inputs."""

    # Validate inputs
    validated_data, errors = validate_inputs(input_data)

    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        # Prepare data for prediction
        dtest = xgb.DMatrix([validated_data])  # Wrap it in a list to make it 2D

        # Make prediction
        predictions = cell_TP_model.predict(dtest)

        # Return floored predictions (adjust rounding if needed)
        results["predictions"] = np.int16(predictions).tolist()

    return results

if __name__ == "__main__":

    # data_in = {'dteday': ['2012-11-6'], 'season': ['winter'], 'hr': ['6pm'], 'holiday': ['No'], 'weekday': ['Tue'],
    #            'workingday': ['Yes'], 'weathersit': ['Clear'], 'temp': [16], 'atemp': [17.5], 'hum': [30], 'windspeed': [10]}
    
    data_in = [1.2, 2.3, 3.1, 4.8, 5.0, 6]
    results = make_prediction(input_data = data_in)
    print(results)