
from typing import List

from fastapi import Query
from pydantic import BaseModel, validator


class PredictionPayload(BaseModel):
    input_data: List = []
