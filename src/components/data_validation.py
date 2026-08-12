import os
import sys
from dataclasses import dataclass

import pandas as pd

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataValidationConfig:
    expected_columns: list = None

    def __post_init__(self):
        self.expected_columns = [
            'ID', 'Year_Birth', 'Education', 'Marital_Status', 'Income',
            'Kidhome', 'Teenhome', 'Dt_Customer', 'Recency', 'MntWines',
            'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
            'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
            'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1',
            'AcceptedCmp2', 'Complain', 'Z_CostContact', 'Z_Revenue', 'Response'
        ]


class DataValidation:
    def __init__(self):
        self.validation_config = DataValidationConfig()

    def validate_columns(self, df: pd.DataFrame) -> bool:
        """Check no expected columns are missing."""
        missing_cols = [c for c in self.validation_config.expected_columns if c not in df.columns]
        if missing_cols:
            logging.error(f"Missing expected columns: {missing_cols}")
            return False
        return True

    def validate_not_empty(self, df: pd.DataFrame) -> bool:
        """Check the dataframe actually has rows."""
        if df.shape[0] == 0:
            logging.error("Dataframe is empty")
            return False
        return True

    def validate_missing_ratio(self, df: pd.DataFrame, threshold: float = 0.5) -> bool:
        """Fail if any column is missing more than `threshold` fraction of its values."""
        missing_ratios = df.isnull().mean()
        bad_cols = missing_ratios[missing_ratios > threshold]
        if len(bad_cols) > 0:
            logging.error(f"Columns exceeding {threshold*100}% missing: {bad_cols.to_dict()}")
            return False
        return True

    def initiate_data_validation(self, train_path, test_path) -> bool:
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Starting data validation")

            checks = [
                self.validate_columns(train_df),
                self.validate_not_empty(train_df),
                self.validate_missing_ratio(train_df),
            ]

            validation_passed = all(checks)

            if validation_passed:
                logging.info("Data validation passed")
            else:
                logging.error("Data validation failed")

            return validation_passed

        except Exception as e:
            raise CustomException(e, sys)