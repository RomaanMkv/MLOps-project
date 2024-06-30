import numpy
numpy.strings = None
numpy.char = None
import great_expectations as gx
import pandas as pd
from great_expectations.data_context import FileDataContext
import pandas as pd
import hydra
from omegaconf import DictConfig
import os

@hydra.main(config_path="../configs", config_name="config")
def sample_data(cfg: DictConfig) -> None:
    # Read the data file
    data_url = cfg.data.path + cfg.data.raw_data_name + str(cfg.data.version) + cfg.data.data_format

    df = pd.read_csv(data_url)

    # Take a sample
    sample_size = cfg.data.sample_size
    sample_df = df.sample(frac=sample_size, random_state=42).reset_index(drop=True)

    # Create samples folder if it doesn't exist
    sample_folder = cfg.data.sample_folder
    os.makedirs(sample_folder, exist_ok=True)

    # Save the sample to the samples folder
    sample_path = os.path.join(sample_folder, "sample.csv")
    sample_df.to_csv(sample_path, index=False)


class DataValidationException(Exception):
    pass

@hydra.main(config_path="../configs", config_name="config")
def validate_initial_data(cfg: DictConfig) -> None:
    data_path=cfg.gx.data_path
    # Initialize the DataContext
    FileDataContext(project_root_dir = cfg.gx.project_root_dir)
    context = gx.get_context(project_root_dir = cfg.gx.project_root_dir)

    # Add or update the pandas datasource
    ds = context.sources.add_or_update_pandas(name=cfg.gx.datasource_name)

    # Add CSV asset
    da1 = ds.add_csv_asset(
        name=cfg.gx.asset_name,
        filepath_or_buffer=data_path,
    )

    # Build batch request
    batch_request = da1.build_batch_request()

    # Add or update the expectation suite
    context.add_or_update_expectation_suite(cfg.gx.suite_name)

    # Get validator for the batch and expectation suite
    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=cfg.gx.suite_name
    )

    # 1. 'month' feature
    validator.expect_column_values_to_not_be_null('month')
    validator.expect_column_values_to_be_between('month', min_value='2017-01', max_value='2024-06')
    validator.expect_column_values_to_match_regex('month', r'^\d{4}-\d{2}$')

    # 2. 'town' feature
    validator.expect_column_values_to_not_be_null('town')
    validator.expect_column_values_to_be_in_set('town', [
        'ANG MO KIO', 'BEDOK', 'BISHAN', 'BUKIT BATOK', 'BUKIT MERAH', 'BUKIT PANJANG', 'BUKIT TIMAH',
        'CENTRAL AREA', 'CHOA CHU KANG', 'CLEMENTI', 'GEYLANG', 'HOUGANG', 'JURONG EAST', 'JURONG WEST',
        'KALLANG/WHAMPOA', 'MARINE PARADE', 'PASIR RIS', 'PUNGGOL', 'QUEENSTOWN', 'SEMBAWANG', 'SENGKANG',
        'SERANGOON', 'TAMPINES', 'TOA PAYOH', 'WOODLANDS', 'YISHUN'
    ])

    # 3. 'flat_type' feature
    validator.expect_column_values_to_not_be_null('flat_type')
    validator.expect_column_values_to_be_in_set('flat_type', ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION'])

    # 4. 'storey_range' feature
    validator.expect_column_values_to_not_be_null('storey_range')
    validator.expect_column_values_to_be_in_set('storey_range', [
        '01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21', '22 TO 24', '25 TO 27',
        '28 TO 30', '31 TO 33', '34 TO 36', '37 TO 39', '40 TO 42', '43 TO 45', '46 TO 48', '49 TO 51'
    ])

    # 5. 'floor_area_sqm' feature
    validator.expect_column_values_to_not_be_null('floor_area_sqm')
    validator.expect_column_values_to_be_between('floor_area_sqm', min_value=31, max_value=249)
    validator.expect_column_values_to_match_regex('floor_area_sqm', r'^\d+(\.\d+)?$')

    # 6. 'lease_commence_date' feature
    validator.expect_column_values_to_not_be_null('lease_commence_date')
    validator.expect_column_values_to_be_between('lease_commence_date', min_value=1966, max_value=2020)
    validator.expect_column_values_to_match_regex('lease_commence_date', r'^\d{4}$')

    # 7. 'resale_price' feature
    validator.expect_column_values_to_not_be_null('resale_price')
    validator.expect_column_values_to_be_between('resale_price', min_value=140000, max_value=1588000)
    validator.expect_column_pair_values_A_to_be_greater_than_B('resale_price', 'floor_area_sqm')

    # 8. 'flat_model' feature
    validator.expect_column_values_to_not_be_null('flat_model') 
    validator.expect_column_values_to_be_in_set('flat_model', [
        'Model A', 'Improved', 'New Generation', 'DBSS', 'Simplified', 'Apartment', 'Standard', 'Premium Apartment',
        'Maisonette', 'Model A-Maisonette', 'Premium Apartment Loft', 'Type S1', 'Type S2', 'Model A2', '2-room',
        'Terrace', 'Adjoined flat', 'Improved-Maisonette', 'Multi Generation', '3Gen', 'Premium Maisonette'
    ])

    # 9. 'block' feature
    validator.expect_column_values_to_not_be_null('block')

    # Save the expectation suite
    validator.save_expectation_suite()

    # Validate the batch
    results = validator.validate()

    if not results.success:
        failed_expectations = [
            result for result in results['results'] if not result['success']
        ]
        for result in failed_expectations:
            expectation = result['expectation_config']['expectation_type']
            print(f"Expectation {expectation}: FAILURE")
            print(f"Details: {result['result']}")
        raise DataValidationException("Data validation failed for one or more expectations.")

    print("All expectations passed successfully.")
