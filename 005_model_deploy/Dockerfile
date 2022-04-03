FROM ubuntu 

RUN apt-get update && \
    apt-get install --assume-yes --fix-missing python-pip git

FROM python:3.7-slim

# Mount current directory to /app in the container image
VOLUME ./:app/

COPY . /app/

# Change WORKDIR
WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 9999:9999
ENTRYPOINT python flask_api.py 9999

CMD ["flask_api.py"]