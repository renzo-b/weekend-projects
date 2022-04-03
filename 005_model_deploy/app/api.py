
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Optional

from fastapi import FastAPI, Request
from models.models import dummy_model

from .schemas import PredictionPayload

# Define application
app = FastAPI(
    title="Forecasting App",
    description="Forecast the future",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.get("/", tags=['General'])
@construct_response
def _index(request: Request):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictionPayload) -> Dict:
    print("payload is ", payload)

    input_data = payload.input_data

    prediction = dummy_model(x=input_data)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": prediction},
    }
    return response
