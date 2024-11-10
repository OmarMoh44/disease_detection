

# Name of docker image of this app is omar1101/ml_app which on docker hub

from app_ml import app

if __name__ == '__main__':
    app.run(debug=True)