import os
import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
import dill
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

    except Exception as e:
        logging.error("Error")
        raise CustomException(e, sys)


def evaluate_model(X_train, X_test, y_train, y_test, models, param):
    try:
        report = {}
        trained_models = {}

        for model_name, model in models.items():
            para = param[model_name]
            print(model_name)
            print(para)
            print(model)
            # gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring="r2")
            # gs.fit(X_train, y_train)

            # best_model = model.__class__(**gs.best_params_)  # Create a new instance
            # best_model.fit(X_train, y_train)

            if para:  # Use GridSearchCV only if there are hyperparameters to tune
                print("---")
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring="r2")
                gs.fit(X_train, y_train)
                print(f"Best parameters for {model_name}: {gs.best_params_}")
                best_model = gs.best_estimator_
            else:
                print("***")
                best_model = model  # Directly use the model if no parameters to tune
                best_model.fit(X_train, y_train)

            trained_models[model_name] = best_model
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score

        return report, trained_models

    except Exception as e:
        logging.error("ERROR")
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        logging.info("Inside the load_object function")
        with open(file_path, 'rb') as file:
            return dill.load(file)  

    except Exception as e:
        logging.error("ERROR")
        raise CustomException(e, sys)