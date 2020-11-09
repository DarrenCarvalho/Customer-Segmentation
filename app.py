# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 05:24:22 2020

@author: Darren Carvalho
"""
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model_lgb.pkl', 'rb'))
pca = pickle.load(open('pca.pkl', 'rb'))


def feature_create(di):
    di['PCA'] = np.float(pca.transform([[di['Age'], di['Work_Experience'], di['Family_Size']]]))

    if (di['Age'] <= 30):
        di['Age_Bin'] = 1
    elif (di['Age'] > 30 and di['Age'] <= 40):
        di['Age_Bin'] = 2
    elif (di['Age'] > 40 and di['Age'] <= 53):
        di['Age_Bin'] = 3
    else:
        di['Age_Bin'] = 4
    di['Ever'] = int(np.where(di['Ever_Married'] == 0, np.where(di['Spending_Score'] == 1, 1, 0), 0))

    data = pd.DataFrame(np.array(list(di.values())).reshape(1,12), columns=di.keys())

    return data


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    keys = [x for x in request.form.keys()]
    key_vals = dict(zip(keys, int_features))
    #print(key_vals)
    data = feature_create(key_vals)

    val = model.predict(data)
    if (val==1):
        final = "Customer Belongs to segment A"
    elif (val==2):
        final = "Customer Belongs to segment B"
    elif (val==3):
        final = "Customer Belongs to segment C"
    else:
        final = "Customer Belongs to segment D"
    return render_template('index.html', prediction=final)


if __name__ == "__main__":
    app.run()




