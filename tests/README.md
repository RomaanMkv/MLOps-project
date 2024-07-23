# tests

This directory contains the test files. Use this directory to store your unit tests, integration tests, and any other tests necessary to ensure the reliability and correctness of your project.

### `test_run.py`

This file contains tests for the `run` function from the `src.main` module. It utilizes `pytest` and `unittest.mock` to create mock objects and functions.


### `test_sample_data.py`

This file tests the `sample_data` function from the `src.data` module. It ensures that the function correctly handles different sample sizes and saves the sample data to the expected location.


## Running the Tests

To run the tests in this directory, use the following command:

```sh
pytest
```

This command will discover and execute all the test files following the `test_*.py` naming convention.