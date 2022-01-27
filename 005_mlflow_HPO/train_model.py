from datetime import datetime, timedelta

import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

days = 10
data_values = np.linspace(10, 20, days)
datetimes = [datetime(2020, 1, 1) + timedelta(days=x) for x in range(days)]
data = np.column_stack((datetimes, data_values))

df = pd.DataFrame(data, columns=['datetime', 'values']).set_index('datetime')


class DummyModel():
    """dummy model to test MLflow with a custom model"""

    def __init__(self, w):
        self.w = w

    def fit(x):
        pass

    def predict(x):
        pass
