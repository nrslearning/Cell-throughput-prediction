"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
#from sklearn.metrics import mean_squared_error, r2_score

from cell_TP_pred_model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_predictions = 178
    
    print("\n###Sample_input_data[0][0]: ", sample_input_data[0][0])
    print("\n>>>sample_input_data[1][0]:", sample_input_data[1][0])
    # When
    result = make_prediction(input_data = sample_input_data[0][0])

    # Then
    predictions = result.get("predictions")
    print("\n###predictions:", predictions)
    print(predictions[0])
    assert isinstance(predictions, list)
    assert isinstance(predictions[0], int)
    assert result.get("errors") is None
    assert (predictions[0] == expected_predictions)
