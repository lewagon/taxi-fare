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
