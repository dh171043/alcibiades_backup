import time
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent / "finalized_model.pkl"
print(f"[{time.ctime()}] --- Script started.")
model = joblib.load(MODEL_PATH)
print(f":[{time.ctime()}] --- joblib.load() completed")
