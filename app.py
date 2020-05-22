from TaxiFareModel.data import BUCKET_NAME
from flask import Flask
from flask import request, jsonify
from datetime import datetime
import joblib
import pytz
import pandas as pd
from TaxiFareModel.gcp import download_model

app = Flask(__name__)

PATH_TO_MODEL = "data/model.joblib"
NYC_DEFAULT_LAT = 40.7808
NYC_DEFAULT_LNG = -73.9772


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
        "pickup_latitude": float(input["pickup_latitude"]),
        "pickup_longitude": float(input["pickup_longitude"]),
        "dropoff_latitude": float(input["dropoff_latitude"]),
        "dropoff_longitude": float(input["dropoff_longitude"]),
        "passenger_count": float(input["passenger_count"]),
        "pickup_datetime": str(input["pickup_datetime"]),
        "key": str(input["key"])}
    return formated_input


pipeline_def = {'pipeline': joblib.load(PATH_TO_MODEL),
            'from_gcp': False}


@app.route('/set_model', methods=['GET', 'POST'])
def set_model():
    inputs = request.get_json()
    model_dir = inputs["model_directory"]
    print(model_dir)
    pipeline_def["pipeline"] = download_model(model_directory="PipelineTest", rm=True)
    pipeline_def["from_gcp"] = True
    return "Pipeline loaded from gcp defined as new pipeline"


@app.route('/predict_fare', methods=['GET', 'POST'])
def predict_fare():
    inputs = request.get_json()
    if isinstance(inputs, dict):
        inputs = [inputs]
    inputs = [format_input(point) for point in inputs]
    # Convert inputs to dataframe to feed as input to our pipeline
    X = pd.DataFrame(inputs)
    pipeline = pipeline_def["pipeline"]
    print(pipeline_def["from_gcp"])
    results = pipeline.predict(X)
    return {"predictions": list(results)}


@app.route('/')
def index():
    return 'OK'


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)
