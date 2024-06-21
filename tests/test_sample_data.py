import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tempfile
import pytest
from unittest.mock import patch
from omegaconf import OmegaConf
import pandas as pd
from src.data import sample_data

@pytest.fixture
def temp_dirs():
    # Setup temporary directories for input and output
    temp_input_dir = tempfile.TemporaryDirectory()
    temp_output_dir = tempfile.TemporaryDirectory()

    yield temp_input_dir, temp_output_dir

    # Cleanup
    temp_input_dir.cleanup()
    temp_output_dir.cleanup()

@pytest.fixture
def sample_data_cfg(temp_dirs):
    temp_input_dir, temp_output_dir = temp_dirs
    # Create a mock CSV file in the temporary input directory
    mock_data_path = os.path.join(temp_input_dir.name, "test_datav1.csv")
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    df.to_csv(mock_data_path, index=False)

    sample_data_cfg_dict = {
        "data": {
            "path": temp_input_dir.name + "/",
            "raw_data_name": "test_data",
            "version": "v1",
            "data_format": ".csv",
            "sample_size": 0.1,
            "sample_folder": "data/sample_data"
        }
    }

    return OmegaConf.create(sample_data_cfg_dict)

@pytest.mark.usefixtures("temp_dirs", "sample_data_cfg")
def test_sample_data_saves_sample(sample_data_cfg):
    # Call sample_data with the configured settings
    sample_data(sample_data_cfg)

    # Check that the sample was saved to the expected location
    expected_sample_path = os.path.join(sample_data_cfg.data.sample_folder, "sample.csv")
    assert os.path.exists(expected_sample_path), f"Expected sample.csv to be created but found nothing."

@patch("subprocess.run")
@pytest.mark.usefixtures("temp_dirs", "sample_data_cfg")
def test_sample_data_calls_dvc_add(mock_subprocess_run, sample_data_cfg):
    # Call sample_data with the configured settings
    sample_data(sample_data_cfg)

    args, kwargs = mock_subprocess_run.call_args[0], mock_subprocess_run.call_args[1]
    expected_command = ["dvc", "add", os.path.join(sample_data_cfg.data.sample_folder, "sample.csv")]
    assert args[0] == expected_command, "Expected dvc add command to be called with the correct arguments."


if __name__ == "__main__":
    pytest.main()
