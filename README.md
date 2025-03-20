# Sentiment Classifier MLOPS

## Requirements:

We need to write a **Python Based CLI** that should be called using **Github Actions** 

### Job 1: Training Workflow
* Ingests a data set and performs some feature engineering
* Executes a simple classification model training script.
* Logs parameters and metrics using tool of your choice.
* Registers the resulting model using the tool of your choice.

### Job 2: Inference Workflow
* Loads data and model
* Performs inference
* Saves inference results

## Deliverables
* Github repo: code, documentation and tests
* Databricks: Jobs, Experiments and Model Registry


## Tools Used:
* Databricks free tier enabled with Unity Catalog
* Github Actions free tier
* Databricks integrated MLFlow


## Overview Architecture

![CI/CD Architecture](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/images/Sentiment%20Analysis%20MLOPS.drawio.png)

### Workflows


#### Create ML Data Workflow

##### Description:
This workflow runs a notebook which takes a raw csv, and based on the user inputs can create train, test and holdout splits with stratification. It also has a feature to version your datasets if needed.

##### Usage
In this workflow, the workflow expects raw data to be saved in the Workspace folder. In our case, we have chosen `/Workspace/data`. It expects the user to follow the following steps before proceeding to run the workflow:
* Make a directory: `/Workspace/data/{DATASET_NAME}/raw/{DATA}.csv`: You can either upload the file through databrick workspace or use the UI to upload the files.
* In the `/Workspace/jobs/run_create_data.json`, you have a workflow job created which points to a notebook. You can choose to use this workflow to perform the tasks mentioned in the Description.

##### Parameters
* notebook_path (str): This is a required field. Either choose the notebook path already present or you can create your own notebook.
* data_name (str): This is a required field. Name of the dataset. This is also used to access the raw dataset in `/Workspace/data/{DATASET_NAME}/raw/{DATA}.csv`.
* data_version (str): This is a required field. In the {DATASET_NAME} folder it creates an additional version folder to main the versioning and source of the dataset.
* test_size (float): This is a required field. The size of the test dataset. 
* holdout (bool): This is an optional field. It creates a third split called holdout.
* holdout_size (float): This is an optional field which is required if holdout is set to `True`. It gives the holdout size. For example, if `test_size` is 0.2 and `holdout_size` is 0.5, the `train_size` is determined with what's remaining which is 0.3  
* stratify (str(List[])): This could be None or pandas columns. It's used for equal distribution of target labels in each of the split.

##### Examples

* The example json to update parameters can be found [here](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/jobs/run_create_data.json)
* The example notebook that the above json runs can be found [here](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/notebooks/create_data.ipynb)

#### Create Train Workflow

##### Description
This workflow runs a notebook which is utilized to train, log and register a machine learning model. It assumes that the `Create ML Data` workflow is run to create the splits because this notebook will utilize the data. In this notebook, we have given an example of binary sentiment analysis which utilized a TFIDFVectorizer for feature engineering. It follows the following steps:
* Read notebook parameters
* Retrieve data based on name and version of the data that we stored before.
* Preprocess/Clean text
* Feature Engineering
* Run Model Pipeline

##### Usage
This notebook consists of a Class called ModelPipeline which is enabled for three Sklearn models: Naive Bayes (Multinomial), Random Forest and Logistic Regression and has the ability to be extended to other models. Given the MODEL_NAME, MODEL_TYPE and MODEL_PARAMS, it trains the model on train and test split of the data. Let's look at the parameters to this notebook.

##### Parameters
* data_name (str): This is a required field. Name of the dataset. This is also used to access the raw dataset in `/Workspace/data/{DATASET_NAME}/raw/{DATA}.csv`.
* data_version (str): This is a required field. In the {DATASET_NAME} folder it creates an additional version folder to main the versioning and source of the dataset.
* model_name (str): This is a required field. This field is used across databricks workspaces to log and register the model and create experiments.
* model_type (str): This is a required field. It either `naive_bayes`, `random_forest` or `logistic_regression` as inputs.
* model_params (str(Dict[str, ])): This is required field. Please look at the respective sklearn model pages for model params.

##### Examples
* The example json to update parameters can be found [here](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/jobs/run_train_model.json)
* The example notebook that the above json runs can be found [here](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/notebooks/train_models.ipynb)

#### Create Inference Workflow

##### Desciption
This workflow runs a notebook which creates an inference on the pretrained model and saves the result in `/Workspace/inference` folder. It goes through following steps:
* Read notebook parameters
* Retrieve holdout/test data
* Perform consistent feature engineering as training notebook.
* Inferences the logged model based on `run_id`
* Saves results in `/Workspace/inference`

##### Usage
This is a simple notebook which given a data_name, data_version and run_id of the model returns data inference.

##### Paramters
* run_id (str): run_id of the model
* data_name (str): This is a required field. Name of the dataset. This is also used to access the raw dataset in `/Workspace/data/{DATASET_NAME}/raw/{DATA}.csv`.
* data_version (str): This is a required field. In the {DATASET_NAME} folder it creates an additional version folder to main the versioning and source of the dataset.

##### Examples
* The example json to update parameters can be found [here](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/jobs/run_inference_model.json)
* The example notebook that the above json runs can be found [here](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/jobs/run_inference_model.json)

#### References:
* https://docs.databricks.com/aws/en/repos/ci-cd-techniques-with-repos
* https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis/code
* https://www.youtube.com/watch?v=f2XQMFod8kg&t=1928s
* https://mlflow.org/docs/latest/api_reference/python_api/mlflow.config.html
* https://docs.databricks.com/aws/en/catalogs/
* https://docs.databricks.com/aws/en/machine-learning/manage-model-lifecycle/
* https://docs.databricks.com/api/workspace/jobs/submit
* https://docs.databricks.com/aws/en/mlflow/end-to-end-example
* https://docs.databricks.com/aws/en/mlflow/models
