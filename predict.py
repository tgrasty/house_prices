# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 18:38:01 2020

@author: grast
"""

import requests
import pandas as pd

train_transformed = pd.read_csv("train_transformed.csv").drop(['Unnamed: 0'],axis=1)
train_trans_sub = train_transformed.loc[0:10]

url = 'http://localhost:5000/predict'
r = requests.post(url,json=train_trans_sub.to_json())

print(r.json())