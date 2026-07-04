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
            
            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            if best_model_score < 0.6:
                raise custom_execption("No best model found")

            logging.info(f"Best model found on train and test data")

            best_model=models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square=r2_score(y_test,predicted)

            return r2_square
            
        except Exception as e:
            raise custom_execption(e,sys)