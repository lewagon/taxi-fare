from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
  'scikit-learn==0.20.2',
  'pandas==0.24.0',
  'google-cloud-storage==1.25.0',  
  'gcsfs']

setup(name='TaxiFareModel',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Taxi Fare Prediction Pipeline')