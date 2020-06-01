from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pytz
import seaborn as sns
import streamlit as st

from TaxiFareModel.gcp import download_model
from TaxiFareModel.utils import geocoder_here
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 14
plt.figure(figsize=(12, 5))
palette = sns.color_palette('Paired', 10)
# UPDATE_DATE = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from TaxiFareModel.data import get_data

st.markdown("# ML Project")
st.markdown("**Taxifare data explorer**")


@st.cache
def read_data(n_rows=10000):
    df = get_data(n_rows=n_rows, local=False)
    return df


def format_input(pickup, dropoff, passengers=1):
    pickup_datetime = datetime.utcnow().replace(tzinfo=pytz.timezone('America/New_York'))
    formated_input = {
        "pickup_latitude": pickup["latitude"],
        "pickup_longitude": pickup["longitude"],
        "dropoff_latitude": dropoff["latitude"],
        "dropoff_longitude": dropoff["longitude"],
        "passenger_count": passengers,
        "pickup_datetime": str(pickup_datetime),
        "key": str(pickup_datetime)}
    return formated_input


def qcut_target(df):
    # We can also visualise binned fare_amount variable
    df['fare-bin'] = pd.cut(df['fare_amount'], bins=list(range(0, 50, 5))).astype(str)
    # Uppermost bin
    df.loc[df['fare-bin'] == 'nan', 'fare-bin'] = '[45+]'
    # Adjust bin so the sorting is correct
    df.loc[df['fare-bin'] == '(5, 10]', 'fare-bin'] = '(05, 10]'
    return df


def main():
    analysis = st.sidebar.selectbox("chose restitution", ["prediction", "Dataviz"])
    if analysis == "Dataviz":
        st.header("TaxiFare Basic Data Visualisation")
        dd = qcut_target(df.copy())
        sns.catplot(x="fare-bin", kind="count", palette=palette, data=dd, height=5, aspect=3)
        st.pyplot()

    if analysis == "prediction":
        st.header("TaxiFare Model predictions")
        # dot = pipeline2dot(pipeline, df)
        # RenderJsDot(dot)
        # apikey_gmaps = 'AIzaSyD9GUiv-boZqngK5EoO0mCWiVUpfJyMbX0'  # (a valid API key is needed to customize map styles)
        # gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13, apikey=apikey)
        # st.pyplot()
        pickup_adress = st.text_input("pickup adress", "251 Church St, New York, NY 10013")
        dropoff_adress = st.text_input("dropoff adress", "434 6th Ave, New York, NY 10011")
        pickup_coords = geocoder_here(pickup_adress)
        dropoff_coords = geocoder_here(dropoff_adress)
        passenger_counts = st.selectbox("# passengers", [1, 2, 3, 4, 5, 6], 1)
        data = pd.DataFrame([pickup_coords, dropoff_coords])
        to_predict = [format_input(pickup=pickup_coords, dropoff=dropoff_coords, passengers=passenger_counts)]
        X = pd.DataFrame(to_predict)
        res = pipeline.predict(X)
        st.write("💸 taxi fare", res[0])
        st.map(data=data)


# print(colored(proc.sf_query, "blue"))
# proc.test_execute()
if __name__ == "__main__":
    pipeline = download_model()
    df = read_data()
    main()
