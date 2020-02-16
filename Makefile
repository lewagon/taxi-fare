PROJECT_ID=TODO
BUCKET_NAME=TODO

run_locally:
	@python -W ignore -m ${PACKAGE_NAME}.${FILENAME}

install_requirements:
	@pip install -r requirements.txt

set_project:
	-@gcloud config set project ${PROJECT_ID}


# ----------------------------------
#         TRAINING
# ----------------------------------
PACKAGE_NAME=TaxiFareModel
FILENAME=trainer
JOB_NAME=taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')
REGION=europe-west1
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15
FRAMEWORK=scikit-learn

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/trainings  \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs 

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
		--python-version ${PYTHON_VERSION} \
		--runtime-version ${RUNTIME_VERSION} \
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
