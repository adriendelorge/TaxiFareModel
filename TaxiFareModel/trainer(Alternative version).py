from sqlite3 import paramstyle
import pandas as pd
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder, Optimizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from TaxiFareModel.utils import compute_rmse
import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
import joblib
from google.cloud import storage

df = get_data()
df = clean_data(df)

y = df.pop("fare_amount")
X = df

X, X_test, y, y_test = train_test_split(X, y, test_size=0.2)

MLFLOW_URI = "https://mlflow.lewagon.ai/"

experiment_name = "[ID] [Bali] [adriendelorge] TaxiFareModel v.uno"

STORAGE_LOCATION = 'models/TaxiFareModel/model.joblib'

BUCKET_NAME = 'adrien_wagon_883'

class Trainer():
    def __init__(self,X,y):
        """
            X: pandas DataFrame
            y: pandas Series
        """

        self.pipeline = None
        self.X = X
        self.y = y

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        dist_pipe = Pipeline([('dist_trans', DistanceTransformer()),('stdscaler', StandardScaler())])

        time_pipe = Pipeline([
            ('time_enc', TimeFeaturesEncoder('pickup_datetime')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ])

        preproc_pipe = ColumnTransformer([
            ('distance', dist_pipe, ["pickup_latitude", "pickup_longitude", 'dropoff_latitude', 'dropoff_longitude']),
            ('time', time_pipe, ['pickup_datetime'])
        ], remainder="drop")


        pipe = Pipeline([
            ('preproc', preproc_pipe),
            ('optimizer', Optimizer()),
            ('linear_model', LinearRegression())
        ])

        self.pipeline=pipe

        return pipe

    def run(self):
        """set and train the pipeline"""

        self.pipeline=Trainer(X,y).set_pipeline()

        params={'linear_model__fit_intercept':[True,False], 'linear_model__normalize':[True,False], 'linear_model__n_jobs':[-1,0,1]}
        grid_search = GridSearchCV(self.pipeline,param_grid=params,cv=5,scoring="r2",n_jobs=-1)

        grid_search.fit(X,y)

        self.pipeline = grid_search.best_estimator_

        return self.pipeline

    def evaluate(self,X_test,y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(X_test)
        rmse = compute_rmse(y_pred,y_test)
        self.mlflow_log_param('model', 'linear')
        self.mlflow_log_metric('rmse', rmse)

        return rmse



    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        self.experiment_name=experiment_name
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def upload_model_to_gcp(self):

        client = storage.Client()

        bucket = client.bucket(BUCKET_NAME)

        blob = bucket.blob(STORAGE_LOCATION)

        blob.upload_from_filename('model.joblib')

    def save_model(self):
        """ Save the trained model into a model.joblib file """
        joblib.dump(self.pipeline, 'model.joblib')

        self.upload_model_to_gcp()


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate


    mytrainer = Trainer(X,y)
        #df = get_data()
        #df = clean_data(df)
        #y = df.pop("fare_amount")
        #X = df
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    mytrainer.set_pipeline()
    mytrainer.run()
    print (mytrainer.evaluate(X_test,y_test))
    mytrainer.save_model()
