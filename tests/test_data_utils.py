import os
import pytest
import pandas as pd
from utils.data import DataPipeline, retrieve_data


def test_data_pipeline(): 
    pipeline = DataPipeline(name='test', version='v1', holdout=True, test_size=0.2, holdout_size=0.2, stratify=['Sentiment'])
    pipeline.read()
    assert not pipeline.df.empty, "Data should be read successfully."
    
    pipeline.split()
    assert 'split' in pipeline.df.columns, "Split column should be created."
    
    assert not pipeline.get_train_data().empty, "Train data should exist."
    assert not pipeline.get_test_data().empty, "Test data should exist."
    assert not pipeline.get_holdout_data().empty, "Holdout data should exist."

def test_invalid_split():
    pipeline = DataPipeline(name='test', version='v1', holdout=True, test_size=0.6, holdout_size=0.5, stratify=['Sentiment'])
    pipeline.read()
    
    with pytest.raises(ValueError, match="holdout_size and test_size combined must be less than 1.0"):
        pipeline.split()

def test_retrieve_data():
    df = retrieve_data('test', 'v1')
    assert not df.empty, "Retrieved data should not be empty."
    assert 'split' in df.columns, "Retrieved data should contain 'split' column."
