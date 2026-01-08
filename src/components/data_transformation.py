import sys
import os 
from dataclasses import dataclass
import pandas as pd 
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preproscessor_obj_file_path: str=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is responsible for data transformation
        1. Handling missing values
        2. Scaling the numerical features   
        3. Encoding the categorical features
        
        '''

        try:
            numerical_features= ['writing_score', 'reading_score']
            categorical_features= [
                'gender',
                'race_ethnicity',  
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            numerical_pipeline= Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy= 'median')), #handles missing numerical values with median
                    ('scaler', StandardScaler(with_mean=False)) #scale the numerical features
                ]
            )

            categorical_pipeline= Pipeline(
                steps= [
                    ('imputer', SimpleImputer(strategy= 'most_frequent')), #handles missing categorical values with most frequent
                    ('one_hot_encoder', OneHotEncoder()), #convert categorical variables into a numerical format
                    ('scaler', StandardScaler(with_mean=False)) #scale the one-hot encoded features
                ]
            )

            
            logging.info(f"Numerical columns scaling completed: {numerical_features}")
            logging.info(f"Categorical columns encoding completed: {categorical_features}")

            preprocessor= ColumnTransformer(
                [
                    ('numerical_pipeline', numerical_pipeline, numerical_features),
                    ('categorical_pipeline', categorical_pipeline, categorical_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            logging.info("Read the train and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj= self.get_data_transformer_object()
            target_column_name= "math_score"
            numerical_features= ['writing_score', 'reading_score']
            
            input_festure_train_df= train_df.drop(columns= [target_column_name], axis=1)
            target_feature_train_df= train_df[target_column_name]

            input_festure_test_df= test_df.drop(columns= [target_column_name], axis=1)
            target_feature_test_df= test_df[target_column_name]

            logging.info(f"Applying preprocessor object on training and testing dataframes")

            input_feature_train_preprocess_obj= preprocessor_obj.fit_transform(input_festure_train_df)
            input_feature_test_preprocess_obj= preprocessor_obj.transform(input_festure_test_df)


            train_arr= np.c_[input_feature_train_preprocess_obj, np.array(target_feature_train_df)]
            test_arr= np.c_[input_feature_test_preprocess_obj, np.array(target_feature_test_df)]

            logging.info("Saved preprocessor object") 

            save_object(
                file_path= self.data_transformation_config.preproscessor_obj_file_path,
                obj= preprocessor_obj
            )   

            return(train_arr, 
                   test_arr, 
                   self.data_transformation_config.preproscessor_obj_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)