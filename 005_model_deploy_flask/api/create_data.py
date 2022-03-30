import os

from schema import ModelPrediction, db

# Connection and table creation
db.connect()
db.create_tables([ModelPrediction])

dataset_prediction = ModelPrediction.create(prediction=200.0,)

dataset_prediction.save()
