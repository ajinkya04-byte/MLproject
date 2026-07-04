from sklearn.model_selection import GridSearchCV
from logging import exception
import os
import sys
from src.logger import logging
from src.exception import custom_execption
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import dill
import numpy as np 
import pandas as pd
import pickle

def save_object(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file:
            dill.dump(obj,file)
    except Exception as e:
        raise custom_execption(e,sys)

def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}

        for i in range(len(list(models.values()))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)


            train_model_score=r2_score(y_train,y_train_pred)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]]=(test_model_score,gs.best_params_)

        return report

    except Exception as e:
        raise custom_execption(e,sys) 
