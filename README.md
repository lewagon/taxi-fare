# Taxi Fare deployment package

You have the structure and a sample code for deploying a custom pipeline to production  

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
2. The "AI Platform Training & Prediction" + "Compute Engine API" on the console for your project
3. A bucket on Google Cloud storage containing a file `data/data_taxi_trips_train_sample_set.csv`
4. You are logged in (`gcloud auth login`) and you've set the project (`gcloud config set project PROJECT_ID`)

## Clone this repo and enter to branch solution

In your terminal, run:

```bash
mkdir ~/code/lewagon && cd $_
git clone git@github.com:lewagon/taxi-fare.git
mv taxi-fare taxi-fare-deployment
cd taxi-fare-deployment
git checkout solution
stt # Open the project in Sublime Text!
```

## Structure of the project
```
├── Makefile          => all necessary commands
├── README.md
├── TaxiFareModel     => python package to be deployed and run on GCP
│   ├── __init__.py
│   ├── data.py
│   ├── encoders.py
│   ├── trainer.py    => main file that will be run by GCP
│   └── utils.py
├── predict.py        => Script to get predictions from our deployed model
├── predictor.py      => Custom prediction class
├── requirements.txt
└── setup.py          => file to specify dependencies and packe info for deployment
```

## Install correct python dependencies

```bash
pip install -r requirements.txt
```

## Adapt code to your settings
### Python code

Replace variable inside `trainer.py` and `data.py` with your variables:
- `PATH_INSIDE_BUCKET` and `BUCKET_NAME` in `data.py` for downlaoding/uploading data from your Storage
- `MODEL_DIRECTY` which indicates where to store your `model.joblib` file

### Makefile Code
Do the same inside Makefile with:
- `PROJECT_ID`, `BUCKET_NAME` for submitting training tasks
- `MODEL_NAME`, and `VERSION_NAME` for deployment
- `PATH_TO_MODEL` which should be the same as `MODEL_DIRECTY` in trainer.py 

## Deploy first test model

Ensure that the code runs locally:
```bash
make run_locally
```

Submit training to GCP:
```sql
make gcp_submit_training 
```

Deploy your model by creating a version you named in the Makefile:
- Make sure you have created a [model](https://console.cloud.google.com/ai-platform/models?project=wagon-bootcamp-256316) first, then run:
```bash
make create_pipeline_version
```

And there you go, you just deployed your model  

**NB**: you trained on a few samples, just for you to check that workflow is funcionnal, you would have to train on more samples in the future

## Use your model
Inspect `predict.py`, replace variables to get data from your storage and test that GCP correctly returns predictions:
```bash
python predict.py
```

## Use it to improve your model
- Train on more samples
- chose more paramters for hyperparameter tuning
- add Custom preprocessing
