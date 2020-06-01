import time
import herepy

import numpy as np

HERE_API_KEY = 'hYjanuWQ92oW2KK_Um_1mmNuR7jW14th3hst9jNO_sc'  # SIGN in to HERE FIRST for freemium plan (no need for Credit card)


def haversine_vectorized(df,
                         start_lat="pickup_latitude",
                         start_lon="pickup_longitude",
                         end_lat="dropoff_latitude",
                         end_lon="dropoff_longitude"):
    """
        Calculate the great circle distance between two points 
        on the earth (specified in decimal degrees).
        Vectorized version of the haversine distance for pandas df
        Computes distance in kms
    """

    lat_1_rad, lon_1_rad = np.radians(df[start_lat].astype(float)), np.radians(df[start_lon].astype(float))
    lat_2_rad, lon_2_rad = np.radians(df[end_lat].astype(float)), np.radians(df[end_lon].astype(float))
    dlon = lon_2_rad - lon_1_rad
    dlat = lat_2_rad - lat_1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_1_rad) * np.cos(lat_2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371 * c


def minkowski_distance(df, p,
                       start_lat="pickup_latitude",
                       start_lon="pickup_longitude",
                       end_lat="dropoff_latitude",
                       end_lon="dropoff_longitude"):
    x1 = df[start_lon]
    x2 = df[end_lon]
    y1 = df[start_lat]
    y2 = df[end_lat]
    return ((abs(x2 - x1) ** p) + (abs(y2 - y1)) ** p) ** (1 / p)


def compute_rmse(y_pred, y_true):
    return np.sqrt(((y_pred - y_true) ** 2).mean())


###################
#  GEOCODER HERE  #
###################
def reverse_geocoder_here(coords, token=HERE_API_KEY):
    """
    coords: (lat, lng)
    ==> 4 Av du General de Gaulle
    """
    geocoderReverseApi = herepy.GeocoderReverseApi(token)
    res = geocoderReverseApi.retrieve_addresses(coords)
    res = res.as_dict()
    adress = res["Response"]["View"][0]["Result"][0]["Location"]["Address"]
    return adress


def geocoder_here(adress, token=HERE_API_KEY):
    """
    adress: 4 Av du General de Gaulle
     ==>  {'Latitude': 48.85395, 'Longitude': 2.27758}
    """
    geocoderApi = herepy.GeocoderApi(api_key=token)
    res = geocoderApi.free_form(adress)
    res = res.as_dict()
    coords = res["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]
    coords = {k.lower(): v for k, v in coords.items()}
    return coords


################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
