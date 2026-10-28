from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from datetime import time
from predictor import load_model_and_encodings, make_prediction

async def lifespan(app: FastAPI):
    print(">> Application Starting")
    print(">> Loading ML assets...")

    app.state.model, app.state.encodings = load_model_and_encodings()
    print(">> Assets loaded")

    yield
    print(">> Application shutting down")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Airline Delay API is live"}

@app.get("/predict/delays")
def predict_delay_endpoint(
    arrival_airport: str, 
    departure_time: time, 
    arrival_time: time
    ):
    if not hasattr(app.state, 'model') or not hasattr(app.state, 'encodings'):
        raise HTTPException(status_code=503,detail = "Model or encodings not loaded")
    try: 
        prediction_minutes = make_prediction(
            model = app.state.model,
            airport_encodings=app.state.encodings,
            arrival_airport=arrival_airport,
            departure_time=departure_time,
            arrival_time=arrival_time
        )
    except KeyError as e:
        raise HTTPException(status_code = 404, detail = str(e))
    
    response_data = {
        "query": {
            "arrival_airport": arrival_airport,
            "local_departure_time": departure_time.isoformat(),
            "local_arrival_time": arrival_time.isoformat()
        },
        "predicted_departure_delay_minutes": round(prediction_minutes, 2)

    }

    return response_data

