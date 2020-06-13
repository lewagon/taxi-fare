from datetime import datetime

import joblib
import pandas as pd
import pytz
from flask import Flask
from flask import request
from flask_cors import CORS
from termcolor import colored

from TaxiFareModel.gcp import download_model

app = Flask(__name__)
CORS(app)

PATH_TO_MODEL = "data/model.joblib"
NYC_DEFAULT_LAT = 40.7808
NYC_DEFAULT_LNG = -73.9772

COLS = ['key',
        'pickup_datetime',
        'pickup_longitude',
        'pickup_latitude',
        'dropoff_longitude',
        'dropoff_latitude',
        'passenger_count']


def format_input(input):
    pickup_datetime = datetime.utcnow().replace(tzinfo=pytz.timezone('America/New_York'))
    default_params = {
        "pickup_latitude": NYC_DEFAULT_LAT,
        "pickup_longitude": NYC_DEFAULT_LNG,
        "pickup_datetime": str(pickup_datetime),
        "key": str(pickup_datetime)}
    for k, v in default_params.items():
        input.setdefault(k, v)
    formated_input = {
        "key": str(input["key"]),
        "pickup_datetime": str(input["pickup_datetime"]),
        "pickup_longitude": float(input["pickup_longitude"]),
        "pickup_latitude": float(input["pickup_latitude"]),
        "dropoff_longitude": float(input["dropoff_longitude"]),
        "dropoff_latitude": float(input["dropoff_latitude"]),
        "passenger_count": float(input["passenger_count"])}
    return formated_input


pipeline_def = {'pipeline': joblib.load(PATH_TO_MODEL),
                'from_gcp': False}


@app.route('/set_model', methods=['GET', 'POST'])
def set_model():
    inputs = request.get_json()
    print(colored("changes model", "green"))
    if isinstance(inputs, dict):
        model_dir = inputs["model_directory"]
    else:
        model_dir = "xgboost_721448_3.8450000286102295"
    print(colored("changes model", "blue"))
    pipeline_def["pipeline"] = download_model(model_directory=model_dir, rm=True)
    pipeline_def["from_gcp"] = True
    print(colored("changes model", "yellow"))
    return {"model": model_dir}


@app.route('/')
def index():
    return 'OK'


@app.route('/predict_fare', methods=['GET', 'POST'])
def predict_fare():
    """
    Expected input
        {"pickup_datetime": 2012-12-03 13:10:00 UTC,
        "pickup_latitude": 40.747,
        "pickup_longitude": -73.989,
        "dropoff_latitude": 40.802,
        "dropoff_longitude":  -73.956,
        "passenger_count": 2}
    :return: {"predictions": [18.345]}
    """
    inputs = request.get_json()
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs = [format_input(point) for point in inputs]
    # Here wee need to convert inputs to dataframe to feed as input to our pipeline
    # Indeed our pipeline expects a dataframe as input
    X = pd.DataFrame(inputs)
    # Here we specify the right column order
    X = X[COLS]
    pipeline = pipeline_def["pipeline"]
    results = pipeline.predict(X)
    results = [round(float(r), 3) for r in results]
    return {"predictions": results}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
