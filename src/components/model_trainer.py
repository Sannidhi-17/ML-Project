import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear regression": LinearRegression(),
                "K-NN": KNeighborsRegressor(),
                "XGBOOST": XGBRegressor(),
                "catBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error"]
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [ .1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8,16,32,64, 128, 256]
                },
                "Linear regression": {
                    # "fit_intercept": [True, False]
                },
                "K-NN":{
                    'n_neighbors':[5,7,9,11]
                },
                "XGBOOST":{
                    'learning_rate': [ .1, .01, .05, .001],
                    'n_estimators': [8,16,32,64, 128, 256]
                },
                "catBoosting Classifier":{
                    'depth': [ 6,9,10],
                    'learning_rate': [ .1, .01, .05, .001]
                },
                "AdaBoost Classifier":{
                    'learning_rate': [ .1, .01, .05, .001],
                    'n_estimators': [8,16,32,64, 128, 256]
                }
            }

            model_report, trained_models = evaluate_model(X_train = X_train, 
                                            y_train=y_train,
                                            X_test=X_test, 
                                            y_test=y_test, 
                                            models = models,
                                            param=params
                                            )

            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            ## To get best model name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            print("x-x-x-")
            print(best_model_name)
            best_model = trained_models[best_model_name]
            logging.info(f"best_model: {best_model}")
            
            if best_model_score < 0.6:
                logging.info("No Best Model Is Found")
                raise CustomException("No best model")
            logging.info("Best model is found for train and test model")

            ## save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                ) 

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            logging.error("ERROR")
            raise CustomException(e, sys)
