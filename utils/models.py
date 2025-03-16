import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env


import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision, recall, f1_score

import time
import cloudpickle
import numpy as np
from typing import Dict, Any

from utils.data import DataPipeline


# The predict method of sklearn's RandomForestClassifier returns a binary classification (0 or 1). 
# The following code creates a wrapper function, SklearnModelWrapper, that uses 
# the predict_proba method to return the probability that the observation belongs to each class. 

class ModelType():
    def __init__(self, model_type):
        models = {
            'random_forest': RandomForestClassifier,
            'logistic_regression': LogisticRegression,
            'svm': SVC,
            'naive_bayes': MultinomialNB,
        }
        return models[model_type]

class SklearnModelWrapper(mlflow.pyfunc.PythonModel):
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]


class ModelPipeline():
    def __init__(self, model_name, model, signature=None, conda_env=None):
        self.conda_env =  _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=["cloudpickle=={}".format(cloudpickle.__version__), "scikit-learn=={}".format(sklearn.__version__)],
            additional_conda_channels=None,
            )

    def find_best_model():
        pass

    def register_model():
        pass

    def train(DataPipeline: DataPipeline, model_name: str, model_type: str, params: Dict[str, Any]):
        with mlflow.start_run(run_name=model_name):
            model_shell = ModelType(model_type)
            model = model_shell(**params)
            
            X_train, y_train = DataPipeline.get_train_data()
            X_test, y_test = DataPipeline.get_test_data()
            
            model.fit(X_train, y_train)
            
            # predict_proba returns [prob_negative, prob_positive], so slice the output with [:, 1]
            predictions_test = model.predict_proba(X_test)[:,1]
            
            auc_score = roc_auc_score(y_test, predictions_test)
            precision_score = precision_score(y_test, predictions_test.round(3))
            recall_score = recall_score(y_test, predictions_test.round(3))
            f1_score = f1_score(y_test, predictions_test.round(3))

            mlflow.log_param(**params)
            
            # Use the area under the ROC curve as a metric.
            mlflow.log_metric({
                'auc_score': auc_score,
                'precision_score': precision_score,
                'recall_score': recall_score,
                'f1_score': f1_score
            })

            wrappedModel = SklearnModelWrapper(model)
            # Log the model with a signature that defines the schema of the model's inputs and outputs. 
            # When the model is deployed, this signature will be used to validate inputs.
            signature = infer_signature(X_train, wrappedModel.predict(None, X_train))
            
            # MLflow contains utilities to create a conda environment used to serve models.
            # The necessary dependencies are added to a conda.yaml file which is logged along with the model.

            mlflow.pyfunc.log_model(model_name, python_model=wrappedModel, conda_env=conda_env, signature=signature)
    
    def inference(DataPipeline: DataPipeline, model):
        pass

