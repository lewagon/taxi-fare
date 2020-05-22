PROJECT_ID=wagon-bootcamp-256316
BUCKET_NAME=wagon-ml-bizot-27

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

run_locally:
	@python -W ignore -m ${PACKAGE_NAME}.${FILENAME}

install_requirements:
	@pip install -r requirements.txt

install:
	@pip install . -U

set_project:
	-@gcloud config set project ${PROJECT_ID}

# ----------------------------------
#         API COMMANDS
# ----------------------------------
APP_NAME=api-wagon
api:
	-@python app.py

heroku_login:
	-@heroku login

heroku_create_app:
	-@heroku create ${APP_NAME}

deploy_heroku:
	-@git push heroku api_ml:master
	-@heroku ps:scale web=1
	-@heroku config:set GOOGLE_APPLICATION_CREDENTIALS="$(< /Users/jbizot/Documents/wagon-bootcamp-256316-51756099e206.json)"


# ----------------------------------
#         TRAINING
# ----------------------------------
PACKAGE_NAME=TaxiFareModel
FILENAME=trainer
JOB_NAME=taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')
REGION=europe-west1
MACHINE_TYPE=n1-standard-4

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/trainings  \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version 3.7 \
		--runtime-version 1.15 \
		--region europe-west1 \
		--scale-tier CUSTOM \
		--master-machine-type ${MACHINE_TYPE}

gcp_test_multiple_trainings:
	@$(MAKE) gcp_submit_training MACHINE_TYPE=n1-standard-8
	@$(MAKE) gcp_submit_training MACHINE_TYPE=n1-standard-16
	@$(MAKE) gcp_submit_training MACHINE_TYPE=n1-standard-32
	#@$(MAKE) gcp_submit_training MACHINE_TYPE=n1-standard-64

# ----------------------------------
#        DEPLOYMENT
# ----------------------------------
MODEL_NAME=TODO
VERSION_NAME=TODO
PATH_TO_MODEL=TODO #same as MODEL_DIRECTY

build_dep:
	-@python setup.py sdist --formats=gztar

upload_dep:
	-@gsutil cp ./dist/${PACKAGE_NAME}-1.0.tar.gz gs://${BUCKET_NAME}/trainings/code/${PACKAGE_NAME}-1.0.tar.gz

create_pipeline_version: build_dep upload_dep
	-@gcloud beta ai-platform versions create ${VERSION_NAME} \
		--model ${MODEL_NAME} \
		--origin gs://${BUCKET_NAME}/models/taxi_fare_model/${PATH_TO_MODEL} \
		--python-version 3.7 \
		--runtime-version 1.15 \
		--prediction-class predictor.Predictor \
    	--package-uris gs://${BUCKET_NAME}/trainings/code/${PACKAGE_NAME}-1.0.tar.gz

# ----------------------------------
#    CLEANING COMMAND
# ----------------------------------


clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr ${PACKAGE_NAME}-*.dist-info
	@rm -fr ${PACKAGE_NAME}.egg-info
	-@rm model.joblib
