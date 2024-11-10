from flask import Flask
from os import environ

app = Flask(__name__)
# app.config['SECRET_KEY'] = environ.get('SECRET_KEY', 'my_secret')
environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from app_ml import routes