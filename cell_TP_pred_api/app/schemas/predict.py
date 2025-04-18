from typing import Any, List, Optional, Union
import datetime

from pydantic import BaseModel

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[int]]
    #predictions: Optional[int]


class DataInputSchema(BaseModel):
    s1_conn_est: List[float]

class MultipleDataInputs(BaseModel):
    inputs: List[DataInputSchema]
