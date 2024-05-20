import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn       # mlflow provides support for every model

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split



def get_data():
    path = "D:\WORK\PYTHON\mlflow\dataset\wine.csv"
    return pd.read_csv(path, sep=";")


def model_evaluation_reg(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

def model_evaluation_cls(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision =  precision_score(y_test, y_pred, average="micro")
    return accuracy, precision


def main():
    # data ingestion
    df = get_data()
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:,:-1], 
        df.iloc[:,-1], 
        test_size=0.2, 
        random_state=42
    )

    # model training and prediction
    '''lr = ElasticNet()
    lr.fit(X_train,y_train)
    y_pred = lr.predict(X_test)'''
    
    rf = RandomForestClassifier(bootstrap=True, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # model evaluation
    #mae, mse, r2 = model_evaluation_reg(y_test, y_pred)
    accuracy, precision = model_evaluation_cls(y_test, y_pred)

    # output
    print("\nMODEL EVALUATIONS")
    print(50*"-")
    print(f"Accuracy : {accuracy*100} %")
    print(f"Precision : {precision*100} %\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e