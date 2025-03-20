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

![CI/CD Architecture](https://github.com/yashad-samant/sentiment_classifier_mlops/blob/dev/images/Screenshot%202025-03-19%20at%209.58.54%E2%80%AFPM.png)

### Workflows

#### Create ML Data Workflow

##### Description:
This workflow takes a raw csv, and based on the user inputs can create train, test and holdout splits with stratification. It also has a feature to version your datasets if needed.

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