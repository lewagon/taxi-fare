import multiprocessing
import os
import time
import warnings

import category_encoders as ce
import mlflow
import pandas as pd
from google.cloud import storage
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from termcolor import colored
from xgboost import XGBRegressor

from TaxiFareModel.data import get_data, clean_df, BUCKET_NAME, DIST_ARGS
from TaxiFareModel.encoders import TimeFeaturesEncoder, DistanceTransformer, AddGeohash, OptimizeSize
from TaxiFareModel.utils import compute_rmse, simple_time_tracker

MODEL_DIRECTY = "PipelineTest"  # must the same as PATH_TO_MODEL inside Makefile
MLFLOW_URI = "http://35.210.166.253:5000"


class Trainer(object):
    TRAINING_NROWS = 10000
    ESTIMATOR = "Linear"
    EXPERIMENT_NAME = "TaxifareModel"

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.nrows = kwargs.get("nrows", self.TRAINING_NROWS)  # nb of rows to train on
        self.grid = kwargs.get("gridsearch", False)  # apply gridsearch if True
        self.local = kwargs.get("local", True)  # if True training is done locally
        self.optimize = kwargs.get("optimize", False)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)  # if True log info to nlflow
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.model_params = None  # for
        self.X_train = X
        self.y_train = y
        self.split = self.kwargs.get("split", False)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(X_train, y_train, test_size=0.15)
        self.log_kwargs_params()
        self.log_machine_specs()

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
        elif estimator == "xgboost":
            model = XGBRegressor(objective='reg:squarederror')
            self.model_params = {'max_depth': range(10, 20, 2),
                                 'n_estimators': range(60, 220, 40),
                                 'learning_rate': [0.1, 0.01, 0.05]
                                 }
        else:
            model = Lasso()
        estimator_params = self.kwargs.get("estimator_params", {})
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def set_pipeline(self):

        pipe_time_features = make_pipeline(TimeFeaturesEncoder(time_column='pickup_datetime'),
                                           OneHotEncoder(handle_unknown='ignore'))
        pipe_distance = make_pipeline(DistanceTransformer(**DIST_ARGS), StandardScaler())
        pipe_geohash = make_pipeline(AddGeohash(), ce.HashingEncoder())

        features_encoder = ColumnTransformer([
            ('distance', pipe_distance, list(DIST_ARGS.values())),
            ('time_features', pipe_time_features, ['pickup_datetime']),
            ('geohash', pipe_geohash, list(DIST_ARGS.values()))
        ], n_jobs=None, remainder="drop")

        self.pipeline = Pipeline(steps=[
            ('features', features_encoder),
            ('rgs', self.get_estimator())])

        if self.optimize:
            self.pipeline.steps.insert(-1, ['optimize_size', OptimizeSize(verbose=False)])

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
                                           n_iter=10,
                                           cv=2,
                                           verbose=1,
                                           random_state=42,
                                           n_jobs=None)

    @simple_time_tracker
    def train(self, gridsearch=False):
        tic = time.time()
        self.set_pipeline()
        if gridsearch:
            self.add_grid_search()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))

    def evaluate(self):
        rmse_train = self.compute_rmse(self.X_train, self.y_train)
        self.mlflow_log_metric("rmse_train", rmse_train)
        if self.split:
            rmse_val = self.compute_rmse(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("rmse_val", rmse_val)
            print(colored("rmse train: {} || rmse val: {}".format(rmse_train, rmse_val), "blue"))
        else:
            print(colored("rmse train: {}".format(rmse_train), "blue"))

    def compute_rmse(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(5), "blue"))
        rmse = compute_rmse(y_pred, y_test)
        return round(rmse, 3)

    def save_model(self, upload=True, auto_remove=True):
        """Save the model into a .joblib and upload it on Google Storage /models folder
        HINTS : use sklearn.joblib (or jbolib) libraries and google-cloud-storage"""
        from sklearn.externals import joblib
        local_model_name = 'model.joblib'
        joblib.dump(self.pipeline, local_model_name)
        print(colored("model.joblib saved locally", "green"))

        if not self.local:
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

    ### MLFlow methods
    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    experiment = "TaxifareModel"
    params = dict(nrows=5000,
                  upload=False,
                  split=True,  # set to False when training no whole training data for final step
                  local=True,  # set to False to get data from GCP (Storage or BigQuery)
                  gridsearch=False,
                  optimize=True,
                  estimator="xgboost",
                  mlflow=True,  # set to True to log params to mlflow
                  experiment_name=experiment)
    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean_df(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)
    print(X_train.memory_usage().sum())
    # Train and save model, locally and
    t = Trainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()
