# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:06:17 2020

@author: chakrabo
"""
#import dependencies
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import h5py
import os


def train(model):
    
    raw_df = pd.read_excel('ED_Data_MegaTable.xlsx', sheet_name='ED_Data_MegaTable')
    raw_df.head()

    #Changing pandas dataframe to numpy array
    X = raw_df.iloc[:,:8].values
    y = raw_df.select_dtypes(include=[object])

    #Normalizing the data
    sc = StandardScaler()
    X = sc.fit_transform(X)

    #one-hot-encoding the class
    ohe = OneHotEncoder()
    y = ohe.fit_transform(y).toarray()

    #test-train split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
    
    #training model
    history = model.fit(X_train, y_train, epochs=100, batch_size=64)
    
    #predict on test data
    y_pred = model.predict(X_test)
    
    #Converting predictions to label
    pred = list()
    for i in range(len(y_pred)):
        pred.append(np.argmax(y_pred[i]))
        
    #Converting one hot encoded test label to label
    test = list()
    for i in range(len(y_test)):
        test.append(np.argmax(y_test[i]))
        
    #print the accuracy 
    a = accuracy_score(pred,test)
    print('Accuracy is:', a*100)
    
    #run the model with validation_data
    history = model.fit(X_train, y_train,validation_data = (X_test,y_test), epochs=100, batch_size=64)
    
    #plot for model accuracy ...
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("model_accuracy.png", bbox_inches='tight', dpi=600)
    plt.show()
    
    #plot of the model loss ...
    plt.plot(history.history['loss']) 
    plt.plot(history.history['val_loss']) 
    plt.title('Model loss') 
    plt.ylabel('Loss') 
    plt.xlabel('Epoch') 
    plt.legend(['Train', 'Test'], loc='upper left') 
    plt.savefig('model_loss.png', bbox_inches='tight', dpi=600)
    plt.show()
    
    pred = pd.DataFrame(y_pred, columns=['below_200', '200_to_250', '250_to_300', '300_to_350', 'above_350', 'err']).to_csv('result.csv')
    
    print("End")
    
    #save model
    model.save("QuenchModel.h5")
    



def model():
    
    # Neural network
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(6, activation='softmax'))

    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #model summary 
    model.summary()

    return model

    
    
def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    quench_model = model()
    train(quench_model)



if __name__== "__main__":
    main()










