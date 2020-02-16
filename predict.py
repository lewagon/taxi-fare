from math import sqrt

import googleapiclient.discovery
import joblib
import pandas as pd
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error

BUCKET_NAME = "wagon-ml-bizot-27"
PATH_INSIDE_BUCKET = "data/data_10Mill.csv"


def predict_json(project, model, instances, version=None):
    """Send json data to a deployed model for prediction. """

    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def get_data(nrows=10000):
    """method to get the training data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    client = storage.Client()
    path = "gs://{}/{}".format(BUCKET_NAME, PATH_INSIDE_BUCKET)
    df = pd.read_csv(path, nrows=nrows)
    return df


def convert_to_json_instances(X_test):
    """
        format of instances to feed to google
        [
            {
                'trip_key': '58e04730-86da-4ca7-890c-4306c4d335ca',
                'pickup_datetime': '2010-04-30 19:37:00 UTC',
                'pickup_longitude': -73.965907,
                'pickup_latitude': 40.752942,
                'dropoff_longitude': -73.965907,
                'dropoff_latitude': 40.752942,
                'passenger_count': 1
            }
        ]
     NB HERE we are in a real use case where we want our data to contain only above fileds
     All feature Engineering is done inside ou pipeline
    """
    return X_test.to_dict(orient="records")


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


if __name__ == '__main__':
    MODEL_AIP = 'taxifare'
    PIPELINE_VERSION_NAME_AIP = "pipelinev1"

    df = get_data(nrows=100)
    # Here we clean df in order to avoid NaN values
    df = df.dropna(how='any',  axis='rows')

    instances = convert_to_json_instances(df.drop("fare_amount", axis=1))
    results = predict_json(project='wagon-bootcamp-256316',
                           model=MODEL_AIP,
                           version=PIPELINE_VERSION_NAME_AIP,
                           instances=instances)

    df["fare_predicted"] = results
    print(evaluate_model(df.fare_amount, df.fare_predicted))
