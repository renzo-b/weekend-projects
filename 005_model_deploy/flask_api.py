import sys

from flask import Flask

app = Flask(__name__)


@app.route("/train_model")
def train_model():
    accuracy = 99

    return {
        "status": "success",
        "selected_model": "model_1",
        "accuracy": f"{accuracy}",
    }


@app.route("/test_model")
def test_model():

    prediction = [1, 2, 3]
    return {
        "status": "success",
        "predictions": prediction,
    }


if __name__ == "__main__":
    try:
        port = int(sys.argv[1])
    except Exception as e:
        port = 80

    print("PORT ", port)

    app.run(host="0.0.0.0", port=port)
