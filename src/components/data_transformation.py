import os
import sys

import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import MyException
from src.logger import logging
from src.utils import save_object

from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifact","preprocessor.pkl")

class DataTransformation: 
    def __init__(self):
        self.data_transformation=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsibe for data transformation.
        '''

        try:
            numerical_features = ['duration', 'days_left'] 
            categorical_features = ['airline', 'source_city', 'departure_time', 'stops', 'arrival_time', 'destination_city', 'class']

            num_pipeline= Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
            )    
            
            logging.info("Numerical columns standard scaling completed.")


            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("ordinal_encoder",OrdinalEncoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Categorical columns encoding completed.")

            logging.info(f"Numerical colums: {numerical_features}")
            logging.info(f"categorical features: {categorical_features}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise MyException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name='price'
            numerical_features = ['duration', 'days_left'] 

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
            )

        except Exception as e:
            raise MyException(e, sys)

