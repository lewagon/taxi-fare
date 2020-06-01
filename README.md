# Taxi Fare Predictor

You have the structure and a sample code for deploying a predictor app on streamlit

The code is very similar to the one you already implemented during week 5  


## 
This package serves two purposes:
- give a package structure to deploy future ML model to production if you have those needs in your carreer as a data scientist
- give a complete solution for the [last exercice](https://kitt.lewagon.com/camps/359/challenges?path=05-Production%2F05-Deploy-to-Production-day2%2F05-Deploy-on-All-data) of week 5:
     - A pipeline with `custom preprocessing` (Custom encoders) 
     - A RandomSearchCV for Hyperparameter tuning
     - A custom predictor class to make our own prdictions
     
## Prerequisite
In the following, we suppose that:
 
1. You have a GCP account, a project, a Service Account key on your disk and its path set up in the `GOOGLE_APPLICATION_CREDENTIALS` env key
2. A bucket on Google Cloud storage containing a previously trained model
3. You are logged in (`gcloud auth login`) and you've set the project (`gcloud config set project PROJECT_ID`)

## Clone this repo and enter to branch streamlit

In your terminal, run:

```bash
mkdir ~/code/lewagon && cd $_
git clone git@github.com:lewagon/taxi-fare.git
mv taxi-fare taxi-fare-deployment
cd taxi-fare-deployment
git checkout streamlit
stt # Open the project in Sublime Text!
```

## Install correct python dependencies

```bash
pip install -r requirements.txt
```

## Adapt code to your settings

Replace variable inside `__init__.py` with your variables:
- `PATH_INSIDE_BUCKET` to download data from storage
- `MODEL_DIRECTY_INSISDE_BUCKET` and `BUCKET_NAME` for uploading and downloading model from GCP Storage


## Trained very simple model locally and upload it to GCP

```bash
make run_locally
```

## Launch your predictor locally
```
make streamlit_local
```

