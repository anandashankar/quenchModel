# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 15:52:35 2020

@author: chakrabo
"""

import os
import pandas as pd
#import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import cv2
import numpy as np
from tensorflow.keras.models import load_model 


def run_file(model):
    
    X = pd.read_excel('ED_Data_MegaTable.xlsx', sheet_name='Table 10')
    
    #predict on test data
    y_pred = model.predict(X)
    
    
    pred = pd.DataFrame(y_pred, columns=['below_200', '200_to_250', '250_to_300', '300_to_350', 'above_350', 'err']).to_csv('prediction_result.csv')
    
    print("End")
    
    

def load_model_():
    
    model = load_model("QuenchModel.h5")

    return model
    
def main():
    
    model = load_model_()
    run_file(model)
    
    
    
if __name__== "__main__":
    main()