import pytest
from fastapi.testclient import TestClient
from main import app
@pytest.fixture(scope="module")
def test_client():
    """Creating a test client that pings my fastapi app"""
    with TestClient(app) as client:
        yield client
    
def test_correct_prediction_input(test_client):
    """ Gives accurate information formatted correctly """
    params = {
        "arrival_airport": "10208",
        "departure_time": "04:30:00",
        "arrival_time": "19:00:00"
    }

    response = test_client.get("/predict/delays", params = params)
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}" # makes sure I got the expected code
    
    response_data = response.json()

    assert "query" in response_data # makes sure the response has a query in it
    assert "predicted_departure_delay_minutes" in response_data # makes sure the response has a predicted_departure_delay_minutes in it
    assert isinstance(response_data["predicted_departure_delay_minutes"], (float, int))

def test_predict_airport_not_found(test_client):
    """  Gives an inaccurate airport code and asserts that the HTTP status code is 404 """
    params = {
        "arrival_airport": "99999",
        "departure_time": "04:30:00",
        "arrival_time": "19:00:00"
    }
    response = test_client.get("/predict/delays", params = params)

    assert response.status_code == 404, f"Expected status code 404, got {response.status_code}" # makes sure I got the expected code
    response_data = response.json()
    assert "not found" in response_data["detail"]


def test_predict_invalid_time_format(test_client):
    """  Gives an inaccurate time formats and asserts that the HTTP status code is 400 """

    params = {
        "arrival_airport": "10208",
        "departure_time": "14:30",  # missing seconds
        "arrival_time": "5:45 PM"   # invalid format
    }

    response = test_client.get("/predict/delays", params=params)

    assert response.status_code == 422, f"Expected status code 422, got {response.status_code}" # makes sure I got the expected code
    response_data = response.json()
    assert "detail" in response_data # detail key is in the response data
    assert isinstance(response_data["detail"], list) # detail is a list