import sys
import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer  
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        ''' 
        This function is responsible for data transformation
        '''

        try:
            numerical_columns = ['writing score',
                                'reading score'
                                ]
            categorical_columns = ['gender',
                                    'race/ethnicity',
                                    'parental level of education',
                                    'lunch',
                                    'test preparation course'
                                    ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical columns encoding completed")
            logging.info("Numerical columns encoding completed")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipline, categorical_columns)
                ]
             )

            return preprocessor
        except Exception as e:
            logging.error("Error")
            raise CustomException(e, sys)



    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtainingbpreprocessing object")

            # preprocessor_obj = self.data_transformation_config
            preprocessor_obj = self.get_data_transformer_object()  
            target_column = "math score"
            numerical_columns = ['writing_score',
                                'reading_score'
                                ]

            input_feature_train_df = train_df.drop(columns=[target_column], axis=1)
            traget_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=[target_column], axis=1)
            traget_feature_test_df = test_df[target_column]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe ")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(traget_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(traget_feature_test_df)
            ]

            logging.info("Saved Preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_file_path,
                obj = preprocessor_obj
            )   

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_file_path
            )

        except Exception as e:
            logging.error("Error")
            raise CustomException(e, sys)