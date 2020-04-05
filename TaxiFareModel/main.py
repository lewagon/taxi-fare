import warnings
from pprint import pprint

from termcolor import colored

from TaxiFareModel.data import clean_df, get_data
from TaxiFareModel.trainer import Trainer

default_params = dict(nrows=40000,
                      upload=False,
                      local=True,  # set to False to get data from GCP (Storage or BigQuery)
                      gridsearch=False,
                      optimize=True,
                      estimator="Linear",
                      mlflow=False,  # set to True to log params to mlflow
                      experiment_name="TaxifareModel")


def get_experiment_param(exp='local'):
    new_params = default_params
    if exp == "local":
        pass
    elif exp == "gcp_machine_types":
        new_params.update(dict(experiment="GCP_Instances",
                               mlflow=True,
                               upload=True,
                               local=False,
                               estimator="RandomForest"))
    else:
        new_params = default_params
    return new_params


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    # Get and clean data
    exp = "gcp_machine_types"
    params = get_experiment_param(exp=exp)
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
