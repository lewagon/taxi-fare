from os.path import split

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from TaxiFareModel.data import df_optimized, get_data, clean_df, DIST_ARGS
from TaxiFareModel.utils import haversine_vectorized, minkowski_distance
import pygeohash as gh
import TaxiFareModel

folder_source, _ = split(TaxiFareModel.__file__)

class CustomEncoder(BaseEstimator, TransformerMixin):
    def size_optimize(self, df):
        return df_optimized(df)


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    # class TimeFeaturesEncoder(CustomEncoder):

    def __init__(self, time_column, time_zone_name='America/New_York'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["dow"] = X.index.weekday
        X["hour"] = X.index.hour
        X["month"] = X.index.month
        X["year"] = X.index.year
        return X[["dow", "hour", "month", "year"]].reset_index(drop=True)

    def fit(self, X, y=None):
        return self


class AddGeohash(BaseEstimator, TransformerMixin):

    def __init__(self, precision=6):
        self.precision = precision

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        X['geohash_pickup'] = X.apply(
            lambda x: gh.encode(x.pickup_latitude, x.pickup_longitude, precision=self.precision), axis=1)
        X['geohash_dropoff'] = X.apply(
            lambda x: gh.encode(x.dropoff_latitude, x.dropoff_longitude, precision=self.precision), axis=1)
        return X[['geohash_pickup', 'geohash_dropoff']]


class DistanceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, distance_type="euclidian", **kwargs):
        self.distance_type = distance_type

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        if self.distance_type == "haversine":
            X["distance"] = haversine_vectorized(X, **DIST_ARGS)
        if self.distance_type == "euclidian":
            X["distance"] = minkowski_distance(X, p=2, **DIST_ARGS)
        if self.distance_type == "manhattan":
            X["distance"] = minkowski_distance(X, p=1, **DIST_ARGS)
        return X[["distance"]]

    def fit(self, X, y=None):
        return self


class AddWeatherData(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        df_weather = pd.read_csv(folder_source+"/data/NY_weather_data.csv")
        cols2drop = list(X)
        cols2drop.extend(["STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION", "DATE", "TAVG", "TSUN"])
        X["DATE"] = X["pickup_datetime"].apply(str).str[:10]
        X = X.merge(df_weather, on="DATE", how="left", )
        X = X.drop(cols2drop, axis=1)
        X[list(X.filter(regex="WT"))] =  X.filter(regex="WT").fillna(0)
        return X

    def fit(self, X, y=None):
        return self




class OptimizeSize(BaseEstimator, TransformerMixin):

    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, X, y=None):
        X = pd.DataFrame(X.toarray())
        assert isinstance(X, pd.DataFrame)
        X = df_optimized(X)
        if self.verbose:
            print(X.head())
        return X

    def fit(self, X, y=None):
        return self


class DistanceToCenter(BaseEstimator, TransformerMixin):
    def __init__(self, verbose=False):
        self.verbose = verbose

    def transform(self, X, y=None):
        nyc_center = (40.7141667, -74.0063889)
        X["nyc_lat"], X["nyc_lng"] = nyc_center[0], nyc_center[1]
        args_pickup = dict(start_lat="nyc_lat", start_lon="nyc_lng",
                           end_lat="pickup_latitude", end_lon="pickup_longitude")
        args_dropoff = dict(start_lat="nyc_lat", start_lon="nyc_lng",
                            end_lat="dropoff_latitude", end_lon="dropoff_longitude")
        X['pickup_distance_to_center'] = haversine_vectorized(X, **args_pickup)
        X['dropoff_distance_to_center'] = haversine_vectorized(X, **args_dropoff)
        return X[["pickup_distance_to_center", "dropoff_distance_to_center"]]

    def fit(self, X, y=None):
        return self

class Direction(BaseEstimator, TransformerMixin):
    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def transform(self, X, y=None):
        def calculate_direction(d_lon, d_lat):
            result = np.zeros(len(d_lon))
            l = np.sqrt(d_lon**2 + d_lat**2)
            result[d_lon>0] = (180/np.pi)*np.arcsin(d_lat[d_lon>0]/l[d_lon>0])
            idx = (d_lon<0) & (d_lat>0)
            result[idx] = 180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
            idx = (d_lon<0) & (d_lat<0)
            result[idx] = -180 - (180/np.pi)*np.arcsin(d_lat[idx]/l[idx])
            return result
        X['delta_lon'] = X[self.start_lon] - X[self.end_lon]
        X['delta_lat'] = X[self.start_lat] - X[self.end_lat]
        X['direction'] = calculate_direction(X.delta_lon, X.delta_lat)
        return X[["delta_lon", "delta_lat", "direction"]]

    def fit(self, X, y=None):
        return self

if __name__ == "__main__":
    params = dict(nrows=1000,
                  upload=False,
                  local=True,  # set to False to get data from GCP (Storage or BigQuery)
                  optimize=False)
    df = get_data(**params)
    df = clean_df(df)
    dir = Direction()
    dist_to_center = DistanceToCenter()
    addw = AddWeatherData()
    X = addw.transform(df)
