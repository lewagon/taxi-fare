# Deploy ML API serving predictions
Here you'll be able to deploy a minimal API to return predictions from pretrained model.
     
## Prerequisite
In the following, we suppose that:
 
1. You have previously trained a pipeline
2. You have saved this model either on GCP Cloud Storage or inside `data/` 
3. If your pipeline included custom transformers, they should be present inside  `TaxiFaremodel/encoders.py`
4. you have created an [free heroku account](https://signup.heroku.com/)


## Clone this repo and enter to branch solution

In your terminal, run:

```bash
mkdir ~/code/lewagon && cd $_
git clone git@github.com:lewagon/taxi-fare.git
cd taxi-fare
git checkout api_ml
stt # Open the project in Sublime Text!
```

## Structure of the project
```
├── Makefile        => usual cookbook to easily run reproducible commands
├── Procfile        => file needed for heroku deployment
├── README.md
├── TaxiFareModel   => python package usefull to call function fron app.py and to install custom transformers
│   ├── __init__.py
│   ├── data.py
│   ├── encoders.py  => custom transformers used inside model.joblib
│   ├── gcp.py
│   ├── main.py
│   ├── trainer.py
│   └── utils.py
├── app.py
├── data
│   └── model.joblib
├── jupy
│   ├── Predict.ipynb => notebook to interact easily with your api
├── predict.py
├── predictor.py
├── requirements.txt
└── setup.py
```


## Adapt code to your settings
### Python code

Replace variable inside `trainer.py` and `data.py` with your variables:
- `BUCKET_NAME` in `data.py` for downlaoding models from GCP

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
