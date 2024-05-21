# type: ignore
import pandas as pd 
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import mlflow
import mlflow.sklearn       # mlflow provides support for every model
from mlflow.models.signature import infer_signature

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

import argparse   #argument parser

# data ingestion
def get_data() -> pd.DataFrame:
    path = "D:\WORK\PYTHON\mlflow\dataset\wine.csv"
    return pd.read_csv(path, sep=";")

# regression metrics
def model_evaluation_reg(y_test:pd.Series, y_pred:pd.Series) -> float:
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, mse, r2

# classification metrics
def model_evaluation_cls(y_test:pd.Series, y_pred:pd.Series) -> float:
    accuracy = accuracy_score(y_test, y_pred)
    precision =  precision_score(y_test, y_pred, average="micro")
    return accuracy, precision


#random search cv
def random_search(model:RandomForestClassifier, X_train:pd.DataFrame, y_train:pd.DataFrame) -> RandomForestClassifier:
    try :
        param = {
                'n_estimators' : [100, 200, 250, 300, 500],
                'criterion' : ["gini", "entropy", "log_loss"],
                'max_depth' : [3,5,8,13,20],
                'min_samples_split' : [2,4,7,8],
                'bootstrap' : [True, False],
                'random_state' : [0, 42, 69, 96]
            }

        random_search_model = RandomizedSearchCV(
            estimator=model,
            param_distributions=param,
            n_iter=10,
            cv=5,
            random_state=42
        )
        random_search_model.fit(X_train, y_train)
        print("Best Parameters : ", random_search_model.best_params_)
        
        #log params using mlflow
        log_params(random_search_model.best_params_)
        
        return random_search_model.best_estimator_
    
    except Exception as e:
        raise e


# grid search cv
def hyperparam_tuning(model:RandomForestClassifier, X_train:pd.DataFrame, y_train:pd.Series) -> RandomForestClassifier: 
    try:
        # specifying the params of RandomForestClassifier to be tested upon
        param = {
            'n_estimators' : [100, 200, 250, 300, 500],
            'criterion' : ["gini", "entropy", "log_loss"],
            'max_depth' : [3,5,8,13],
            'min_samples_split' : [2,4,7,8],
            'bootstrap' : [True, False],
            'random_state' : [0, 42, 69, 96]
        }
        # deploying grid search cv
        gs_cv = GridSearchCV(
            estimator=model,
            param_grid=param,
            cv=10,
            n_jobs=-1
        )
        # training grid search cv 
        gs_cv.fit(X_train, y_train)
        # printing best params
        print(gs_cv.best_params_)

        #log params using mlflow
        log_params(gs_cv.best_params_)

        return gs_cv.best_estimator_
        # this function will return the best RandomForestClassifier trained model
    
    except Exception as e:
        raise e

# logging the parameters
def log_params(parameters:dict):
    for key in parameters.keys():
        mlflow.log_param(key,parameters[key])


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

    # logging using mlflow here
    with mlflow.start_run():

        # model training and prediction
        '''lr = ElasticNet()
        lr.fit(X_train,y_train)
        y_pred = lr.predict(X_test)'''
        
        rf = RandomForestClassifier(bootstrap=True, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        signature = infer_signature(X_train, y_pred)
        # log and register model
        mlflow.sklearn.log_model(rf, "pre_tuning_model", signature=signature)

        # model evaluation
        #mae, mse, r2 = model_evaluation_reg(y_test, y_pred)
        accuracy_before, precision_before = model_evaluation_cls(y_test, y_pred)

        # output
        print("\nMODEL EVALUATIONS BEFORE ANY HYPERPARAMETER TUNING")
        print(50*"-")
        print(f"Accuracy : {accuracy_before*100} %")
        print(f"Precision : {precision_before*100} %\n")
        mlflow.log_metric("accuracy_pre_hyperparam_tuning", accuracy_before)
        mlflow.log_metric("precision_pre_hyperparam_tuning", precision_before)

        # using grid search fine tuned model
        rf_tuned= random_search(rf, X_train, y_train)
        y_pred_tuned = rf_tuned.predict(X_test)

        signature = infer_signature(X_train, y_pred_tuned)
        # log and register model
        mlflow.sklearn.log_model(rf_tuned, "post_tuning_model", signature=signature)

        accuracy_after, precision_after = model_evaluation_cls(y_test, y_pred_tuned)
        mlflow.log_metric("accuracy_post_hyperparam_tuning", accuracy_after)
        mlflow.log_metric("precision_post_hyperparam_tuning", precision_after)

        # output
        print("\nMODEL EVALUATIONS AFTER RANDOM SEARCH CV HYPERPARAMETER TUNING")
        print(50*"-")
        print(f"Accuracy : {accuracy_after*100} %")
        print(f"Precision : {precision_after*100} %\n")

# we are getting just 65% accuracy normally
# HYPERPARAMETER TUNING REQUIRED
# even worse results after hyperparameter tuning

# we could've used argparse to test on various hyperparameters but we would be typing in each and every argument, i find the hyperparameter tuning to be a better option
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        raise e