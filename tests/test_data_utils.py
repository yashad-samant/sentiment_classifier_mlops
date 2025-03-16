import os
import random
import pytest
import pandas as pd
from pathlib import Path
from utils.data import DataPipeline

#test_data = Path("/Workspace/data/test/sentimentdataset_sampled.csv")

@pytest.fixture
def sample_csv(tmp_path):
    """Creates a sample CSV file for testing."""
    data = pd.DataFrame({
        'feature1': [random.randint(1, 100) for _ in range(50)],
        'feature2': [random.randint(1, 100) for _ in range(50)],
        'label': [0, 1] * 25
    })
    print(data)
    file_path = tmp_path / "test_data.csv"
    data.to_csv(file_path, index=False)
    return file_path


def test_read_valid_csv(sample_csv):
    """Test if the read method correctly reads a CSV file."""
    data_obj = DataPipeline(file_path=str(sample_csv))
    data_obj.read()
    assert hasattr(data_obj, 'df'), "DataFrame attribute should be created"
    assert not data_obj.df.empty, "DataFrame should not be empty"
    assert list(data_obj.df.columns) == ['feature1', 'feature2', 'label'], "Columns should match the CSV"


def test_read_invalid_file():
    """Test if the read method raises an exception for unsupported file formats."""
    data_obj = DataPipeline(file_path="invalid_file.txt")
    with pytest.raises(Exception, match="Unsupported file format"):
        data_obj.read()


def test_split_without_holdout(sample_csv):
    """Test splitting data without a holdout set."""
    data_obj = DaDataPipelineta(file_path=str(sample_csv), holdout=False, test_size=0.3)
    data_obj.read()
    data_obj.split()

    assert 'split' in data_obj.df.columns, "Split column should exist"
    assert set(data_obj.df['split']) == {'train', 'test'}, "Only train and test splits should be present"
    assert len(data_obj.df[data_obj.df['split'] == 'test']) > 0, "Test split should have data"
    assert len(data_obj.df[data_obj.df['split'] == 'train']) > 0, "Train split should have data"


def test_split_with_holdout(sample_csv):
    """Test splitting data with a holdout set."""
    data_obj = DataPipeline(file_path=str(sample_csv), holdout=True, test_size=0.3, holdout_size=0.4)
    data_obj.read()
    data_obj.split()

    assert 'split' in data_obj.df.columns, "Split column should exist"
    assert set(data_obj.df['split']) == {'train', 'test', 'holdout'}, "Train, test, and holdout splits should be present"
    assert len(data_obj.df[data_obj.df['split'] == 'holdout']) > 0, "Holdout split should have data"


def test_split_with_stratify(sample_csv):
    """Test stratified splitting."""
    data_obj = DataPipeline(file_path=str(sample_csv), holdout=False, test_size=0.2, stratify=['label'])
    data_obj.read()
    data_obj.split()

    train_label_dist = data_obj.df[data_obj.df['split'] == 'train']['label'].value_counts(normalize=True)
    test_label_dist = data_obj.df[data_obj.df['split'] == 'test']['label'].value_counts(normalize=True)

    assert train_label_dist.equals(test_label_dist), "Stratified split should maintain label proportions"

