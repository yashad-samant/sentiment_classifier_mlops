import pytest
import mlflow
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from src.model_pipeline import get_model_type, ModelPipeline  # Adjust import as per your project structure


# Mock MLflow (to avoid real logging during tests)
@pytest.fixture
def mock_mlflow():
    with patch.object(mlflow, "start_run"), patch.object(mlflow, "log_metric"), patch.object(mlflow.sklearn, "log_model"):
        yield


# ✅ **Test 1: Model selection works correctly**
@pytest.mark.parametrize("model_type, expected_class", [
    ("random_forest", "RandomForestClassifier"),
    ("logistic_regression", "LogisticRegression"),
    ("naive_bayes", "MultinomialNB"),
    ("invalid_model", None),
])
def test_get_model_type(model_type, expected_class):
    model = get_model_type(model_type)
    if model:
        assert model.__name__ == expected_class
    else:
        assert model is None


# ✅ **Test 2: Training pipeline runs without errors**
def test_train_pipeline(mock_mlflow):
    # Generate synthetic data
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a ModelPipeline instance
    pipeline = ModelPipeline(model_name="test_model", model_type="random_forest", params={"n_estimators": 10})

    # Run training (ensuring it doesn’t crash)
    pipeline.train(X_train, y_train, X_test, y_test)


# ✅ **Test 3: MLflow logging is called correctly**
def test_mlflow_logging(mock_mlflow):
    # Generate synthetic data
    X, y = make_classification(n_samples=50, n_features=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = ModelPipeline("test_model", "random_forest", {"n_estimators": 5})

    # Mock MLflow functions
    with patch.object(mlflow, "log_metric") as mock_log_metric, \
         patch.object(mlflow.sklearn, "log_model") as mock_log_model:

        pipeline.train(X_train, y_train, X_test, y_test)

        # Check that logging functions were called
        assert mock_log_metric.call_count >= 3  # AUC, precision, recall, etc.
        assert mock_log_model.called


# ✅ **Test 4: Inference works correctly**
@patch("mlflow.pyfunc.load_model")
def test_inference(mock_load_model):
    # Mock MLflow model loading
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1, 0, 1])
    mock_load_model.return_value = mock_model

    pipeline = ModelPipeline("test_model", "random_forest")
    predictions = pipeline.inference(pd.DataFrame([[0.1, 0.2, 0.3]]))  # Fake test data

    assert predictions.tolist() == [1, 0, 1]  # Ensure expected predictions

