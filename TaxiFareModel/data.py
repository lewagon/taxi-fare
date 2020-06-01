import pandas as pd
from google.cloud import storage

from TaxiFareModel import BUCKET_NAME, PATH_INSIDE_BUCKET
from TaxiFareModel.utils import simple_time_tracker

#PATH_INSIDE_BUCKET = "data/data_10Mill.csv"


DIST_ARGS = dict(start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude")


@simple_time_tracker
def get_data(nrows=10000, local=False, optimize=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    if local:
        path = "data/data_data_10Mill.csv"
    else:
        path = "gs://{}/{}".format(BUCKET_NAME, PATH_INSIDE_BUCKET)
    if optimize:
        cols = infer_dtypes(path)
        cols = {k: v for k, v in cols.items() if "latitude" not in k and "longitude" not in k}
        df = pd.read_csv(path, nrows=nrows, dtype=cols)
    else:
        df = pd.read_csv(path, nrows=nrows)
    if "test" in list(df):
        df.drop("test", axis=1, inplace=True)
    return df


def clean_df(df, test=False):
    """ Cleaning Data based on Kaggle test sample
    - remove high fare amount data points
    - keep samples where coordinate wihtin test range
    """
    df = df.dropna(how='any', axis='rows')
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df


def df_optimized(df, verbose=True, **kwargs):
    """
    Reduces size of dataframe by downcasting numeircal columns
    :param df: input dataframe
    :param verbose: print size reduction if set to True
    :param kwargs:
    :return:
    """
    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
            if type == "float":
                df[col] = pd.to_numeric(df[col], downcast="integer")
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100
    GB = out_size / 1000000000
    if verbose:
        print("optimized size by {} % | {} GB".format(ratio, GB))
    return df


def infer_dtypes(path):
    """
    infer optimized dtypes for dataframe future dataframe csv loading
    :param path:
    :return: dict {"colname": dtype} to pass as argument to pd.read_csv
    """
    df = pd.read_csv(path, nrows=100)
    df_opt = df_optimized(df, verbose=False)
    dtypes = df_opt.dtypes
    colnames = dtypes.index
    types = [i.name for i in dtypes.values]
    column_types = dict(zip(colnames, types))
    return column_types


if __name__ == "__main__":
    params = dict(nrows=10000000,
                  upload=False,
                  local=True,  # set to False to get data from GCP (Storage or BigQuery)
                  optimize=True)
    df = get_data(**params)
    params["optimize"] = False
    df_2 = get_data(**params)
    m1 = df.memory_usage().sum()/1000000
    m2 = df_2.memory_usage().sum()/1000000
    print(m1, m2, m1 / m2)
    mm = pd.merge(df, df_2, on="key")
