from flask import Flask,request, url_for, redirect, render_template, jsonify
from pycaret.classification import *
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model('deployment_slide_rotate')
cols = ['diffpress', 'flowrate', 'holedepth', 'hookload', 'pumppress', 'rotrpm',
       'torque', 'wob']

@app.route('/')
def home():
    return 'Website is up running'

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = int(prediction.Label[0])
    return jsonify(output)

if __name__ == '__main__':
    app.run(debug=True)
