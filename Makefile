PACKAGE_NAME=TaxiFareModel

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------

run_locally:
	@python -W ignore -m ${PACKAGE_NAME}.trainer

install_requirements:
	@pip install -r requirements.txt

set_project:
	-@gcloud config set project ${PROJECT_ID}

streamlit_local:
	-@streamlit run TaxiFareModel/streamlit.py

# ----------------------------------
#    LOCAL INSTALL COMMANDS
# ----------------------------------
install: install_requirements
	@pip install . -U


clean:
	@rm -fr */__pycache__
	@rm -fr __init__.py
	@rm -fr build
	@rm -fr dist
	@rm -fr ${PACKAGE_NAME}-*.dist-info
	@rm -fr ${PACKAGE_NAME}.egg-info
	-@rm model.joblib
