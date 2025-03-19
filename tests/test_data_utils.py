import os
import pytest
import pandas as pd
from data import DataPipeline, retrieve_data

data_dir = '/tmp/test_data'
os.makedirs(data_dir, exist_ok=True)

def create_sample_csv(name, data):
    file_path = os.path.join(data_dir, name, 'raw', 'data.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    data.to_csv(file_path, index=False)

def test_data_pipeline():
    # Create sample data
    data = pd.DataFrame({
        'feature1': range(10),
        'feature2': range(10, 20),
        'label': ['A', 'B'] * 5
    })
    create_sample_csv('sample_dataset', data)
    
    pipeline = DataPipeline(name='sample_dataset', version='v1', holdout=True, test_size=0.2, holdout_size=0.2, stratify=['label'])
    pipeline.read()
    assert not pipeline.df.empty, "Data should be read successfully."
    
    pipeline.split()
    assert 'split' in pipeline.df.columns, "Split column should be created."
    
    assert not pipeline.get_train_data().empty, "Train data should exist."
    assert not pipeline.get_test_data().empty, "Test data should exist."
    assert not pipeline.get_holdout_data().empty, "Holdout data should exist."

def test_invalid_split():
    data = pd.DataFrame({
        'feature1': range(10),
        'feature2': range(10, 20),
        'label': ['A', 'B'] * 5
    })
    create_sample_csv('invalid_dataset', data)
    
    pipeline = DataPipeline(name='invalid_dataset', version='v1', holdout=True, test_size=0.6, holdout_size=0.5, stratify=['label'])
    pipeline.read()
    
    with pytest.raises(ValueError, match="holdout_size and test_size combined must be less than 1.0"):
        pipeline.split()

def test_retrieve_data():
    df = retrieve_data('sample_dataset', 'v1')
    assert not df.empty, "Retrieved data should not be empty."
    assert 'split' in df.columns, "Retrieved data should contain 'split' column."
