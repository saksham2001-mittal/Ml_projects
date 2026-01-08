import os 
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

 
@dataclass
class ModelTrainerClassConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")  

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerClassConfig()

    def initate_model_trainer(self, train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models= {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            model_report: dict= evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)

            ## To get the best model score from the dict
            best_model_score= max(sorted(model_report.values()))

            ## To get the best model name from the dict
            best_model_name= list(model_report.keys())[
                list(model_report.values()).index(best_model_score)]

            best_model= models[best_model_name]

            if best_model_score< 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found on both training and testing dataset")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
            predicted_result= best_model.predict(X_test)
            r2_square= r2_score(y_test, predicted_result)
            return r2_square
         
        except Exception as e:
            raise CustomException(e, sys)