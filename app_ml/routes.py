from flask import request, render_template, jsonify
from app_ml.models import numerical_features, min_max, categorical_features, categorical_unique_values, predict_target
from app_ml import app



@app.route("/")
@app.route("/home")
def home():
    models = ['KNN','Logistic regression', 'Keras']
    categories = {}
    for index in range(len(categorical_features)):
        categories = {**categories,categorical_features[index]:categorical_unique_values[index]}
    numerics = {}
    for index in range(len(numerical_features)):
        numerics = {**numerics, numerical_features[index]:min_max[index]}
    return render_template('home.html', numerics=numerics, categories=categories, models=models)


@app.route("/predict", methods=['POST'])
def predict():
    obj = predict_target(request)
    if 'target' in obj:
        target = None
        if obj['target'] == 1:
           target = 'Has heart disease'
        else:
            target = 'Does not have heart disease'
        return render_template('result.html',target=target,error=None)
    else:
        return render_template('result.html',target=None,error=obj['error'])
    
