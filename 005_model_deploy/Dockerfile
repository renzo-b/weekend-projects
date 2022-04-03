FROM python:3.7-slim

WORKDIR 005_model_deploy
# COPY setup.py setup.py
COPY requirements.txt requirements.txt
COPY Makefile Makefile

RUN apt-get update \
    && apt-get install -y 

COPY app app
COPY models models

RUN pip install -r requirements.txt

EXPOSE 5000
ENTRYPOINT ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "5000", "--reload", "--reload-dir", "app"]