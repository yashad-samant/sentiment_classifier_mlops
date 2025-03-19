# importing requred libraries
import os
import mlflow
import mlflow.sklearn
import databricks.connect as db_connect
from mlflow.models.signature import infer_signature


import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

import time
import cloudpickle
import numpy as np
from typing import Dict, Any

from utils.data import DataPipeline

# setting uri
ARTIFACTS_PATH = "artifacts"
mlflow.set_tracking_uri("databricks")


def get_model_type(model_type: str):
    """
    This is a helper function to get the model type based on the model name.
    Documentation on sklearn can be found here:
    1. RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    2. LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    3. MultinomialNB: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

    Args:
        model_type (str): The type of the model. Currently the code supports RandomForestClassifier, LogisticRegression, MultinomialNB.
    Returns:
        sklearn.model
    """
    models = {
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegression,
        'naive_bayes': MultinomialNB,
    }
    return models.get(model_type)


class MLFlowUtils():
    def __init__(self):
        mlflow.login()
        spark_ctx = db_connect.DatabricksSession.builder.serverless(True).getOrCreate()


class ModelPipeline():
    """
    Model Pipeline for train and inference. This is a helper class to run the train and
    inference notebooks. It also contains the train function which is used to train the model and 
    inference function to load the model. We have packaged MLFlow utilities in here but can be decoupled 
    in the next iteration.
    
    Args:
        model_name (str): The name of the model.
        model_type (str): The type of the model. Currently the code supports random_forest, logistic_regression, naive_bayes.
        params (Dict[str, Any]): The parameters for the model. Please follow SKLEARN DOCS to find out the parameters for the
            respective models.
    Returns:
        None
    """
    def __init__(
        self, 
        model_name: str, 
        model_type: str,
        params: Dict[str, Any] = {}
        ):
        self.model_name = model_name
        self.model_type = model_type
        self.params = params

    # TODO: Use MLFLOW API to find best runs
    def find_best_model():
        pass

    #TODO: Break the train function into smaller functions
    def train(self, X_train, y_train, X_test, y_test) -> None:
        """
        Trains, logs and registers the model.

        Args:
            X_train (pd.DataFrame): The training data.
            y_train (pd.DataFrame): The training labels.
            X_test (pd.DataFrame): The test data.
            y_test (pd.DataFrame): The test labels.
        """
        # TODO: need to set tracking and registry uri in yaml files 
        mlflow.set_registry_uri("file:/Workspace")
        
        with mlflow.start_run():
            # gets the Sklearn model class based on the model type.
            model_shell = get_model_type(self.model_type)

            # creates model based on the model type and the parameters.
            model = model_shell(**self.params)
            
            # trains the model given train data and labels.
            model.fit(X_train, y_train)
            
            # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
            predictions_prob = model.predict_proba(X_test)[:,1]
            #TODO: get predictions from model threshold
            predictions = model.predict(X_test)
            
            # compute the precision, recall, f1, and auc metrics.
            auc = roc_auc_score(y_test, predictions_prob)
            p = precision_score(y_test, predictions.round(3), average='macro')
            r = recall_score(y_test, predictions.round(3), average='macro')
            f1 = f1_score(y_test, predictions.round(3), average='macro')

            # log the model metrics for tracking in MLFLOW. 
            mlflow.log_metric("auc", auc)
            mlflow.log_metric("precision", p)
            mlflow.log_metric("recall", r)
            mlflow.log_metric("f1", f1)

            # log the model parameters for tracking if present.
            if len(self.params) > 0:
                for key, value in self.params.items():
                    mlflow.log_param(key, value)

            # get the signature so that it can be used for verification while inferencing.
            signature = infer_signature(X_test, predictions)           
            

            # logs and registers model.
            # by default the yaml will log the model in mlflow experiments and register the model in local folder.
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=ARTIFACTS_PATH,
                registered_model_name=self.model_name,
                signature=signature)
            

    def inference(self, holdout):
        logged_model = 'runs:/b50219b5b91243b6a31dc4c74103a593/model'

        # Load model as a PyFuncModel.
        loaded_model = mlflow.pyfunc.load_model(logged_model)

        # Predict on a Pandas DataFrame.
        print(loaded_model.predict(holdout))

