**Project Overview**

This project aims to build a Random Forest-based binary classification model and deploy it into production using a Python CLI tool. The deployment will integrate Databricks Jobs, Tracking Tools, and GitHub Actions. The model training and inference will be automated with scheduled Databricks jobs, ensuring seamless model updates and inference execution.

**Project Requirements**

**Functional Requirements**

1. Python CLI Tool: 
  * A command-line interface (CLI) written in Python. 
  * The CLI should create two separate Databricks jobs when invoked by a GitHub Action.
  * Object-Oriented Design should be used where applicable.

2. Databricks Jobs:

  * Model Training (Runs every 30 days)
    - Ingests a dataset and performs feature engineering.
    - Trains a Random Forest binary classifier.
    - Logs training parameters and metrics using a tracking tool.
    - Registers the trained model.

  * Model Inference (Runs daily)
    - Loads the trained model and new data.
    - Performs inference.
    - Saves inference results.

3. GitHub Actions Workflow:
   * Automates the execution of the CLI tool to create Databricks jobs.
   * Ensures proper usage of secrets and environment setup.
   * Deploys code to production when merged into the release branch.

4. Production Deployment:
   * The model should be deployed in Databricks with appropriate workspace folder structures.
   * Naming conventions should be used for environment separation (dev, staging, prod).

**Non-Functional Requirements**

1. Scalability: Ensure that the system can handle increasing data loads.
2. Modularity: Code should be modular and reusable.
3. Traceability: Use tracking tools for logging metrics and parameters.
4. Security: Secure secrets and credentials in GitHub Actions.
5. Testing: Ensure the project includes unit tests and integration tests.
6. Documentation: Provide clear documentation on setup, usage, and maintenance.