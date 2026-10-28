import joblib
import json
import numpy as np
from pathlib import Path
from typing import Tuple, Any, Dict
from datetime import time

# setting robust file paths to get around multiple working computers
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "finalized_model.pkl"
ENCODINGS_PATH = BASE_DIR / "airport_encodings.json"

def load_model_and_encodings() -> Tuple[Any, Dict[str, int]]:
    """
    loads model and encodings for future use
    """
    # make sure there isn't a model already loaded
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Encodings file not fount at {MODEL_PATH}")
    if ENCODINGS_PATH.exists():
            with open(ENCODINGS_PATH, "r") as f:
                airport_encodings = json.load(f)
    else:
        raise FileNotFoundError(f"Encodings file not fount at {ENCODINGS_PATH}")

    return model, airport_encodings

def make_prediction(
        model: Any,
        airport_encodings: Dict[str, int],
        arrival_airport: str,
        departure_time: time,
        arrival_time: time
) -> float:
    """
    ensures input data is correctly formatted and makes prediction using loaded model

    returns the predicted departure delay in minutes as a float
    """
    # preprocess data
    arrival_airport_encoded = airport_encodings.get(arrival_airport.upper())
    if arrival_airport_encoded is None:
         raise KeyError(f"Airport code '{arrival_airport}' not found.")
    departure_time_minutes = departure_time.hour * 60 +departure_time.minute
    arrival_time_minutes = arrival_time.hour * 60 + arrival_time.minute

    if arrival_time_minutes < departure_time_minutes:
        # Add a day's worth of minutes
        duration_minutes = (arrival_time_minutes + 24 * 60) - departure_time_minutes
    else:
        duration_minutes = arrival_time_minutes - departure_time_minutes


    num_airports = len(airport_encodings)
    airport_vector = np.zeros(num_airports)
    airport_vector[arrival_airport_encoded] = 1


    features = np.concatenate([
         airport_vector,
         [departure_time_minutes, arrival_time_minutes, duration_minutes]
    ]).reshape(1, -1)
    prediction_minutes = model.predict(features)[0]

    return prediction_minutes