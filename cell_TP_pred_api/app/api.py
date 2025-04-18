import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Body
from fastapi.encoders import jsonable_encoder
from cell_TP_pred_model import __version__ as model_version
from cell_TP_pred_model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    print("####### Health")
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()


example_input = {
    "inputs": [
        {
            "s1_conn_est": [19, 25, 20, 35, 33, 27],
        }
    ]
}

@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.MultipleDataInputs = Body(..., example=example_input)) -> Any:
    """
    Bike rental count prediction with the bikeshare_model
    """
    print("############## Predict")
    input_df = pd.DataFrame(jsonable_encoder(input_data.inputs))
    # Extract the validated list of floats
    #input_list = input_data.s1_conn_est
    print("\n###input_df:", input_df)
    print("\n###type(input_df):", type(input_df))
    print("\n***")
    print("\n***input_df.s1_conn_est:", input_df.s1_conn_est)
    print("\n***type(input_df.s1_conn_est):", type(input_df.s1_conn_est))
    
    s1_list = input_df["s1_conn_est"].iloc[0]
    print("\n>>>>s1_list:",s1_list)
    print("\n>>>>type(s1_list):",type(s1_list))
    
    # Make prediction
    results = make_prediction(input_data=s1_list)
    print(results)
    if results["errors"] is not None:
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))

    return results