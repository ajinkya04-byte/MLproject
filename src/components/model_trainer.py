from ctypes import LittleEndianStructure
import sys
import sys
import os
from dataclasses import dataclass

import numpy as np 
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.exception import custom_execption
from src.logger import logging
from src.utils import save_object,evaluate_model


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Spliting train and test data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            logging.info("Model training")
            
            models={
            "Random Forest":RandomForestRegressor(),
            "Decision Tree":DecisionTreeRegressor(),
            "Gradient Boosting":GradientBoostingRegressor(),
            "AdaBoost":AdaBoostRegressor(),
            "KNN":KNeighborsRegressor(),
            "LinearRegression":LinearRegression(),  
            "XGBoost":XGBRegressor(),
            "CatBoost":CatBoostRegressor(verbose=False)
           }

            params={
                "Decision Tree":{
                    'criterion':['squared_error', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256] 
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,0.01,0.05,0.001],
                    'n_estimators':[8,16,32,64,128,256],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]
                },
                "LinearRegression":{},
                "KNN":{
                    'n_neighbors':[5,7,9,11]
                },
                "XGBoost":{
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[0.1,0.01,0.001,0.05]
                },
                "CatBoost":{
                    'iterations':[30,50,100],
                    'learning_rate':[0.1,0.01,0.001,0.05],
                    'depth':[6,8,10]
                },
                "AdaBoost":{
                    'learning_rate':[0.1,0.01,0.001,0.05],
                    'n_estimators':[8,16,32,64,128,256]
                }
                
           }
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

            
            best_model_name = max(model_report, key=lambda k: model_report[k][0])

            best_model_score = model_report[best_model_name][0]

            best_model = models[best_model_name]

            if (best_model_score) < 0.6:
                raise custom_execption("No best model found")

            logging.info(f"Best model found on train and test data")

            

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)

            return r2_square
            
        except Exception as e:
            raise custom_execption(e,sys)