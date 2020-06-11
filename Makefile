# ----------------------------------
#         LOCAL SET UP
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

install: install_requirements
	@pip install . -U

upload_simple_model:
	@python -m TaxiFareModel.trainer

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

heroku_set_gcp_env:
	-@heroku config:set GOOGLE_APPLICATION_CREDENTIALS="$(< /Users/jbizot/Documents/wagon-bootcamp-256316-51756099e206.json)"

# ----------------------------------
#    CLEANING COMMAND
# ----------------------------------

clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr $TaxiFareModel-*.dist-info
	@rm -fr $TaxiFareModel.egg-info
