import os
from math import sqrt

import googleapiclient.discovery
import joblib
import pandas as pd
from TaxiFareModel.data import get_data

from TaxiFareModel.gcp import download_model
from google.cloud import storage
from sklearn.metrics import mean_absolute_error, mean_squared_error

BUCKET_NAME = "wagon-ml-bizot-27"
PATH_INSIDE_BUCKET = "data/data_10Mill.csv"


#######################
#       FROM GCP      #
#######################
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


#######################
#       FROM GCP      #
#######################

def get_test_data():
    """method to get the training data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    path = "data/test.csv"
    df = pd.read_csv(path)
    return df


def evaluate_model(y, y_pred):
    MAE = round(mean_absolute_error(y, y_pred), 2)
    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
    res = {'MAE': MAE, 'RMSE': RMSE}
    return res


def generate_submission_csv(folder="xgboost_721448_4.019999980926514", kaggle_upload=False):
    df_test = get_test_data()
    X_train = get_data(nrows=100, local=True)
    df_test = df_test[list(X_train.drop("fare_amount", axis=1))]
    if "test" not in list(df_test):
        df_test["test"] = False
    pipeline = download_model(folder)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    df_test["fare_amount"] = y_pred
    df_sample = df_test[["key", "fare_amount"]]
    name = f"data/submissions/predictions_{folder}.csv"
    df_sample.to_csv(name, index=False)
    print("prediction saved under kaggle format")
    if kaggle_upload:
        kaggle_message_submission = name[:-4]
        command = f'kaggle competitions submit -c new-york-city-taxi-fare-prediction -f {name} -m "{kaggle_message_submission}"'
        os.system(command)


def predict_aip():
    MODEL_AIP = 'taxifare'
    PIPELINE_VERSION_NAME_AIP = "pipelinev1"

    df = get_data(nrows=100)
    # Here we clean df in order to avoid NaN values
    df = df.dropna(how='any', axis='rows')

    instances = convert_to_json_instances(df.drop("fare_amount", axis=1))
    results = predict_json(project='wagon-bootcamp-256316',
                           model=MODEL_AIP,
                           version=PIPELINE_VERSION_NAME_AIP,
                           instances=instances)

    df["fare_predicted"] = results
    print(evaluate_model(df.fare_amount, df.fare_predicted))


if __name__ == '__main__':
    folder="xgboost_721448_4.019999980926514"
    folder = "xgboost_360658_4.218999862670898"
    folder = "xgboost_7208704_3.6530001163482666"
    folder = "xgboost_360658_3.8310000896453857"
    #model = download_model(folder)
    generate_submission_csv(folder, kaggle_upload=True)

