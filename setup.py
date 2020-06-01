from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs==0.6.0',
    'pandas==0.24.2',
    'scikit-learn==0.20.4',
    'google-cloud-storage==1.26.0',
    'pygeohash',
    'category_encoders',
    'termcolor',
    'mlflow',
    'xgboost==0.90',
    'memoized_property',
    'psutil']

setup(
    name='TaxiFareModel',
    version='1.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Taxi Fare Prediction Pipeline'
)




