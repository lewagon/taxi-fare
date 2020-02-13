PACKAGE_NAME=TaxiFareModel
FILENAME=trainer
BUCKET_NAME=wagon-ml-bizot-27
PROJECT_ID=wagon-bootcamp-256316
JOB_NAME=taxi_fare_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')
REGION=europe-west1
PYTHON_VERSION=3.7
RUNTIME_VERSION=1.15
FRAMEWORK=scikit-learn


run_locally:
	@python -m ${PACKAGE_NAME}.${FILENAME}

gcp_submit_training: 
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/trainings  \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--stream-logs 

