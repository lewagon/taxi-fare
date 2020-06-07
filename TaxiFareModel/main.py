import warnings
from pprint import pprint

from termcolor import colored

from TaxiFareModel.data import clean_df, get_data
from TaxiFareModel.trainer import Trainer

default_params = dict(nrows=10000,
                      upload=False,
                      local=True,  # set to False to get data from GCP (Storage or BigQuery)
                      gridsearch=False,
                      optimize=False,
                      estimator="xgboost",
                      mlflow=False,  # set to True to log params to mlflow
                      experiment_name="test",
                      pipeline_memory=None,
                      distance_type="manhattan",
                      feateng=["distance_to_center", "direction", "distance", "time_features", "geohash"])


def get_experiment_param(exp='local', nrows=1000000, gridsearch=False):
    new_params = default_params
    params = default_params
    params["experiment_name"] = exp
    if exp in ["local", "test"]:
        return default_params
    elif exp == "gcp_machine_types":
        new_params.update(dict(experiment="GCP_Instances",
                               mlflow=True,
                               upload=True,
                               local=False,
                               estimator="RandomForest"))
    elif exp == "train_scale":
        new_params.update(dict(experiment=exp,
                               nrows=nrows,
                               mlflow=True,
                               upload=True,
                               pipeline_memory=True,
                               gridsearch=gridsearch,
                               local=False,
                               optimize=True,
                               estimator="xgboost"))

    else:
        new_params = default_params
    return new_params


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    exp = "train_scale"
    params = get_experiment_param(exp=exp, gridsearch=False, nrows=10000000)
    params["pipeline_memory"] = True
    params["n_jobs"] = -1
    pprint(params)
    print("############   Loading Data   ############")
    df = get_data(**params)
    df = clean_df(df)
    y_train = df["fare_amount"]
    X_train = df.drop("fare_amount", axis=1)
    print("shape: {}".format(X_train.shape))
    print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
    # Train and save model, locally and
    t = Trainer(X=X_train, y=y_train, **params)
    del X_train, y_train
    print(colored("############  Training model   ############", "red"))
    t.train()
    print(colored("############  Evaluating model ############", "blue"))
    t.evaluate()
    print(colored("############   Saving model    ############", "green"))
    t.save_model()
