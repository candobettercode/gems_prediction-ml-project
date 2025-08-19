# Baisc import
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,       # MAE
    mean_squared_error,        # MSE
    mean_squared_log_error,    # MSLE
    median_absolute_error,     # MedAE
    r2_score,                  # R²
    explained_variance_score,  # EVS
    max_error                   # Max Error
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import catboost
import lightgbm as lgb

from src.exception import MyException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(),
                'Lasso Regression': Lasso(),
                'Random Forest Regressor': RandomForestRegressor(),
                'K-Nearest Neighbors': KNeighborsRegressor(),
                'Gradient Boosting Regressor': GradientBoostingRegressor(),
                'XGBoost Regressor': xgb.XGBRegressor(),
                'CatBoost Regressor': catboost.CatBoostRegressor(verbose=False),
                'LightGBM Regressor': lgb.LGBMRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params={
                "K-Nearest Neighbors": {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
                },

                "Linear Regression": {
                    'fit_intercept': [True, False],
                },

                "Ridge Regression": {
                    'alpha': [0.1, 1.0, 10.0],
                    'fit_intercept': [True, False]
                },

                "Lasso Regression": {
                    'alpha': [0.1, 1.0, 10.0],
                    'fit_intercept': [True, False]
                },

                "Random Forest Regressor":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.8,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "XGBoost Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "CatBoost Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }

            print(f"X_train: {X_train.shape}, y_train: {y_train.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

            model_report:dict=evaluate_models(X_train=X_train, 
                                                X_test=X_test,
                                                y_train=y_train,
                                                y_test=y_test,
                                                models=models,
                                                param=params)
            ## to get the best model score from dict
            best_model_score=max(sorted(model_report.values()))

            ## To get the best modelname from dict  
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise MyException("No best model found")
            
            logging.info(f"Best '{best_model_name}' model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted=best_model.predict(X_test)
            r2_square=r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise MyException(e,sys)



