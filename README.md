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

![CI/CD Architecture Followed]["images/images/Screenshot 2025-03-19 at 9.58.54 PM.png"]