import pytest
from unittest.mock import patch, MagicMock
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.main import run

@pytest.fixture
def mock_config():
    mock_cfg = MagicMock()
    mock_cfg.train_data_version = "v1"
    mock_cfg.test_data_version = "v2"
    return mock_cfg

@patch('src.main.save_best_model')
@patch('src.main.log_metadata')
@patch('src.main.train')
@patch('src.main.load_features')
def test_run(mock_load_features, mock_train, mock_log_metadata, mock_save_best_model, mock_config):
    # Mock return values for load_features
    mock_load_features.side_effect = [
        (MagicMock(), MagicMock()),  # For train
        (MagicMock(), MagicMock())   # For test
    ]

    # Call the function with mock config
    run(mock_config)

    # Check if load_features is called with the correct arguments
    mock_load_features.assert_any_call(name="features_target", version="v1")
    mock_load_features.assert_any_call(name="features_target", version="v2")

    # Check if train is called with correct arguments
    args, kwargs = mock_train.call_args
    assert len(args) == 2  # Two positional arguments (X_train and y_train)
    assert 'cfg' in kwargs
    assert kwargs['cfg'] == mock_config

    # Check if save_best_model is called with the config
    mock_save_best_model.assert_called_once_with(cfg=mock_config)
