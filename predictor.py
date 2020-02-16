import os
import traceback

import pandas as pd
from sklearn.externals import joblib


class Predictor(object):

    def __init__(self, model):
        self._model = model

    def predict(self, instances, **kwargs):
        """
            format of instances (coming from df.to_dict(orient='records'))
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
        """
        try:
            # Important  HERE
            # our trained pipeline needs a dataframe as input
            df = pd.DataFrame(instances)
            predictions = self._model.predict(df)
            return predictions.tolist()
        except BaseException:
            return {"error": True, "traceback": traceback.format_exc()}

    @classmethod
    def from_path(cls, model_dir):
        model_path = os.path.join(model_dir, 'model.joblib')
        model = joblib.load(model_path)
        return cls(model)

