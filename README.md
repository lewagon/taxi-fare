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
- `BUCKET_NAME`, and `PROJECT_ID` in `__init__.py` for downloading models from GCP

### Makefile Code
Do the same inside Makefile with:
- `PROJECT_ID`, `BUCKET_NAME` to get models 

## Deploy app locally

Ensure that the app runs locally:
```bash
python app.py
```

## Deploy to heroku
login to heroku
```bash
make heroku_login
```

create app on heroku
```bash
make heroku_create_app
```

Deploy to heroku
```bash
make deploy_heroku
```

## Test you api
Open `Predict.ipynb` under `jupy` folder and start interrogating your api

