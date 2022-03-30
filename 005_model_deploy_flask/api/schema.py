import peewee as pw

import config

db = pw.PostgresqlDatabase(
    config.POSTGRES_DB,
    user=config.POSTGRES_USER,
    password=config.POSTGRES_PASSWORD,
    host=config.POSTGRES_HOST,
    port=config.POSTGRES_PORT,
)


class BaseModel(pw.Model):
    """Base model to re-use peewee meta class"""

    class Meta:
        database = db


class ModelPrediction(BaseModel):
    prediction = pw.FloatField()

