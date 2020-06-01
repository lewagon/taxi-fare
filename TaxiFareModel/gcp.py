import os

import joblib
from google.cloud import storage
from termcolor import colored

from TaxiFareModel import MODEL_DIRECTY_INSISDE_BUCKET
from TaxiFareModel.data import BUCKET_NAME


def storage_upload(model_directory, bucket=BUCKET_NAME, rm=False):
    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        'taxi_fare_model',
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.upload_from_filename('model.joblib')
    print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                  "green"))
    if rm:
        os.remove('model.joblib')


def download_model(model_directory=MODEL_DIRECTY_INSISDE_BUCKET, bucket=BUCKET_NAME, rm=True):
    # creds = get_credentials()
    # client = storage.Client(credentials=creds, project=PROJECT_ID).bucket(bucket)
    client = storage.Client().bucket(bucket)

    storage_location = '{}/{}/{}/{}'.format(
        'models',
        'taxi_fare_model',
        model_directory,
        'model.joblib')
    blob = client.blob(storage_location)
    blob.download_to_filename('model.joblib')
    print(f'=> pipeline downloaded from storage')
    model = joblib.load('model.joblib')
    if rm:
        os.remove('model.joblib')
    return model
