# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:40:43 2020

@author: grast
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd
import math

def inverse_transform(data_transformed):
  try:
    return round(data_transformed.apply(lambda x: math.exp(x/1e6)-1)).astype(int)
  except:
    return (np.exp(data_transformed/1e6)-1).round().astype(int)

model_save_name = 'house_prices_xgboost.json'
path = "C:/Users/grast/1. Python Code/Kaggle - House Prices/{0}".format(model_save_name)
model = pickle.load(open(path,'rb'))

# app
app = Flask(__name__)

# routes
@app.route('/predict', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    #data.update((x, [y]) for x, y in data.items())
    data_df = pd.read_json(data)

    # predictions
    result = model.predict(data_df)
    result = inverse_transform(pd.DataFrame(result)).to_numpy()

    # send back to browser
    output = {i:int(result[i]) for i in range(0,len(result))}
    #output = {'results': int(result[0])}

    # return data
    return jsonify(results=output)

if __name__ == '__main__':
    app.run(port = 5000, debug=False)