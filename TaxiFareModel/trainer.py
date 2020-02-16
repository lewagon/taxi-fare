import os
import warnings
from termcolor import colored
from google.cloud import storage
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from TaxiFareModel.data import get_data, clean_df, BUCKET_NAME, DIST_ARGS
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash
from TaxiFareModel.utils import compute_rmse, simple_time_tracker

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_DIRECTY = "Pipeline2"  # must the same as PATH_TO_MODEL inside Makefile


class Trainer(object):
    TRAINING_NROWS = 10000
    ESTIMATOR = "Linear"
    SPLIT = False

    def __init__(self, data=None, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param data:
        :param kwargs:
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.nrows = kwargs.get("nrows", self.TRAINING_NROWS)  # cf doc above
        self.df = None
        self.model_params = None  # for
        if isinstance(data, pd.DataFrame):
            self.df = data

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "Lasso":
            model = Lasso()
        elif estimator == "Ridge":
            model = Ridge()
        elif estimator == "Linear":
            model = LinearRegression()
        elif estimator == "GBM":
            model = GradientBoostingRegressor()
        elif estimator == "RandomForest":
            model = RandomForestRegressor()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto', 'sqrt']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        model.set_params(**estimator_params)
        print(model.__class__.__name__)
        return model

    def set_pipeline(self):

        time_features = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),
                                      OneHotEncoder(handle_unknown='ignore'))

        pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())

        features_encoder = ColumnTransformer([
            ('distance', DistanceTransformer(**DIST_ARGS), list(DIST_ARGS.values())),
            ('time_features', time_features, ['pickup_datetime']),
            ('geohash', pipe_geohash, list(DIST_ARGS.values()))
        ])

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('rgs', self.get_estimator())])

    def add_grid_search(self):
        """"
        Apply Gridsearch on self.params defined in get_estimator
        {'rgs__n_estimators': [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
          'rgs__max_features' : ['auto', 'sqrt'],
          'rgs__max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}
        """
        # Here to apply ramdom search to pipeline, need to follow naming "rgs__paramname"
        params = {"rgs__" + k: v for k, v in self.model_params.items()}
        self.pipeline = RandomizedSearchCV(estimator=self.pipeline, param_distributions=params,
                                           n_iter=100, cv=3, verbose=2,
                                           random_state=42, n_jobs=-1)

    @simple_time_tracker
    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)

    def train(self):
        if not isinstance(self.df, pd.DataFrame):
            self.df = get_data(nrows=100000, bq=True)
        y_train = self.df["fare_amount"]
        X_train = self.df.drop("fare_amount", axis=1)
        self.split = self.kwargs.get("split", self.TRAINING_NROWS)  # cf doc above
        if self.split:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
        self.set_pipeline()
        self.add_grid_search()
        self.fit(X_train, y_train)
        rmse_train = self.evaluate(X_train, y_train)
        if self.split:
            rmse_val = self.evaluate(X_val, y_val)
            print(colored("rmse train: {} || rmse val: {}".format(rmse_train, rmse_val), "green"))
        else:
            rmse_val = self.evaluate(X_val, y_val)
            print(colored("rmse train: {}".format(rmse_train), "green"))

    def upload_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        from sklearn.externals import joblib
        local_model_name = 'model.joblib'
        joblib.dump(self.pipeline, local_model_name)
        print("model.joblib saved locally")

        if upload:
            client = storage.Client().bucket(BUCKET_NAME)

            storage_location = '{}/{}/{}/{}'.format(
                'models',
                'taxi_fare_model',
                MODEL_DIRECTY,
                local_model_name)
            blob = client.blob(storage_location)
            blob.upload_from_filename(local_model_name)
            print(colored("=> model.joblib uploaded to bucket {} inside {}".format(BUCKET_NAME, storage_location),
                          "green"))
            if auto_remove:
                os.remove(local_model_name)


if __name__ == "__main__":
    # Get and clean data
    N = 10000
    split = True  # set to False when training no whole training data for final step
    df = get_data(nrows=N)
    df = clean_df(df)

    # Train and save model, locally and
    t = Trainer(data=df, estimator="RandomForest")
    t.train()
    t.upload_model(upload=True)
