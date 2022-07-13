import os
from math import sqrt

import joblib
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

PATH_TO_LOCAL_MODEL = 'model.joblib'

BUCKET_NAME = 'adrien_wagon_883'
BUCKET_TEST_DATA_PATH = 'data/test.csv'


def get_test_data(nrows):
    """method to get the test data (or a portion of it) from google cloud bucket
    To predict we can either obtain predictions from train data or from test data"""
    # Add Client() here
    df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TEST_DATA_PATH}", nrows=nrows)
    #df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
    return df


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


#def evaluate_model(y, y_pred):
#    MAE = round(mean_absolute_error(y, y_pred), 2)
#    RMSE = round(sqrt(mean_squared_error(y, y_pred)), 2)
#    res = {'MAE': MAE, 'RMSE': RMSE}
#    return res


def generate_prediction(nrows):
    df_test = get_test_data(nrows)
    pipeline = joblib.load(PATH_TO_LOCAL_MODEL)
    if "best_estimator_" in dir(pipeline):
        y_pred = pipeline.best_estimator_.predict(df_test)
    else:
        y_pred = pipeline.predict(df_test)
    return y_pred




if __name__ == '__main__':
    print(generate_prediction(10000))
