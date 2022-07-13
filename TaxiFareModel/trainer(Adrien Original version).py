import pandas as pd
from TaxiFareModel.data import get_data, clean_data
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from TaxiFareModel.utils import compute_rmse



class Trainer():
    def __init__(self):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        df = get_data()
        df = clean_data(df)

        y = df.pop("fare_amount")
        X = df

        self.pipeline = None
        self.X = X
        self.y = y

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test

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
            ('linear_model', LinearRegression())
        ])

        self.pipeline=pipe

        return pipe

    def run(self):
        """set and train the pipeline"""

        self.pipeline.fit(self.X_train, self.y_train)

        return self.pipeline

    def evaluate(self):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = compute_rmse(y_pred,self.y_test)
        return rmse


if __name__ == "__main__":
    # get data
    # clean data
    # set X and y
    # hold out
    # train
    # evaluate
    mytrainer = Trainer()
        #df = get_data()
        #df = clean_data(df)
        #y = df.pop("fare_amount")
        #X = df
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    mytrainer.set_pipeline()
    mytrainer.run()
    print (mytrainer.evaluate())
