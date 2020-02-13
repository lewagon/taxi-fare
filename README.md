# Taxi Fare model

The goal is to build a simple model and submit a training job to AI platform.

Then we can AI platform to make **prediction** on new data.

Here are the steps executed by `TaxiFareModel/trainer.py`:

1. Get a sample of training data from Google Cloud Storage Bucket
2. Train the model
3. Upload the model to Google Cloud Storage

## Install Requirements

In the following, we suppose that:

1. You have a GCP account, a project, a Service Account key on your disk and its path set up in the `GOOGLE_APPLICATION_CREDENTIALS` env key
2. The "AI Platform Training & Prediction" + "Compute Engine API" on the console for your project
3. A bucket on Google Cloud storage containing a file `data/data_taxi_trips_train_sample_set.csv`
4. You are logged in (`gcloud auth login`) and you've set the project (`gcloud config set project PROJECT_ID`)

## Clone this repo

In your terminal, run:

```bash
mkdir ~/code/lewagon && cd $_
git clone git@github.com:lewagon/taxi-fare.git
cd taxi-fare
stt # Open the project in Sublime Text!
```

## Check that the code runs locally

In Sublime Text, open the `Makefile` and set the two first lines variables:

- `PROJECT_ID`
- `BUCKET_NAME` (where GCP will store training material)

Then open the `TaxiFareModel/trainer.py` and set the two global variables:

- `BUCKET_NAME` (where the training data is stored)
- `PATH_INSIDE_BUCKET` (should be `data/_____.csv`)

Then launch:

```bash
make run_locally
```

Check that :

- A `model.joblib` file has been created locally
- This file has been uploaded to the bucket `BUCKET_NAME`, in the following path: `models/taxi_fare_model/VERSION`.

## Specify your requirements to GCP inside setup.py

To get the latest versions of the dependencies you can run:

```bash
pip install --upgrade gcsfs google-cloud-storage pandas scikit-learn
```

Check version of python libraries we have installed in the virtualenv:

```bash
pip freeze | grep -E "pandas|scikit|google-cloud-storage|gcsfs"
```

Make sure to put them in the `REQUIRED_PACKAGES` list in `setup.py`.

## Submit Training to GCP

The `Makefile` uses the following configuration:

```
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15
```

Fore more information about latest runtimes, check out the [documentation](https://cloud.google.com/ai-platform/training/docs/runtime-version-list?hl=en).

You can now run:

```bash
make gcp_submit_training
```

:bulb: You can now follow your job submission on the command line or on [AI Platform GCP console](https://console.cloud.google.com/ai-platform/jobs?hl=en)

When your job is finished check on your [Storage Bucket](https://console.cloud.google.com/storage/browser?hl=en) that the `model.joblib` has been updated

