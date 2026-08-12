import os
import sys
import json
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.decomposition import PCA

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("models", "preprocessor.pkl")
    power_transformer_file_path: str = os.path.join("models", "power_transformer.pkl")
    reference_date_file_path: str = os.path.join("models", "reference_date.json")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def impute_income(self, train_df, test_df):
        """Fill missing Income using group medians learned from TRAIN only."""
        group_medians = train_df.groupby(['Education', 'Marital_Status'])['Income'].median()
        overall_median = train_df['Income'].median()

        def fill_income(row):
            if pd.isna(row['Income']):
                key = (row['Education'], row['Marital_Status'])
                return group_medians.get(key, overall_median)
            return row['Income']

        train_df['Income'] = train_df.apply(fill_income, axis=1)
        test_df['Income'] = test_df.apply(fill_income, axis=1)

        return train_df, test_df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Row-wise feature creation."""
        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
        df['Age'] = 2025 - df['Year_Birth']
        df['TotalChildren'] = df['Kidhome'] + df['Teenhome']

        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        df['TotalSpend'] = df[mnt_cols].sum(axis=1)

        purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        df['TotalPurchases'] = df[purchase_cols].sum(axis=1)
        df['DealAffinity'] = df['NumDealsPurchases'] / (df['TotalPurchases'] + 1)

        df['CustomerTenureDays'] = (df['Dt_Customer'].max() - df['Dt_Customer']).dt.days

        campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        df['TotalCampaignsAccepted'] = df[campaign_cols].sum(axis=1)

        drop_cols = ['Kidhome', 'Teenhome', 'ID', 'Z_CostContact', 'Z_Revenue',
                     'TotalPurchases', 'Year_Birth', 'Dt_Customer']

        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df

    def get_preprocessor_object(self, num_cols, cat_cols):
        """Build the ColumnTransformer + PCA pipeline, unfitted."""
        try:
            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop="first"))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, num_cols),
                ("cat_pipeline", cat_pipeline, cat_cols)
            ])

            full_pipeline = Pipeline(steps=[
                ("preprocessor", preprocessor),
                ("pca", PCA(n_components=0.95, random_state=42))
            ])

            return full_pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data")

            train_df, test_df = self.impute_income(train_df, test_df)
            logging.info("Imputed missing Income values")

            # Save the reference date BEFORE engineer_features drops Dt_Customer
            max_date = train_df['Dt_Customer'].pipe(pd.to_datetime, dayfirst=True).max()
            os.makedirs(os.path.dirname(self.transformation_config.reference_date_file_path), exist_ok=True)
            with open(self.transformation_config.reference_date_file_path, "w") as f:
                json.dump({"max_dt_customer": max_date.isoformat()}, f)
            logging.info("Saved reference date for CustomerTenureDays")

            train_df = self.engineer_features(train_df)
            test_df = self.engineer_features(test_df)
            logging.info("Feature engineering completed")

            skewed_cols = ['Income', 'TotalSpend', 'TotalCampaignsAccepted']
            pt = PowerTransformer(method='yeo-johnson')
            train_df[skewed_cols] = pt.fit_transform(train_df[skewed_cols])
            test_df[skewed_cols] = pt.transform(test_df[skewed_cols])

            save_object(
                file_path=self.transformation_config.power_transformer_file_path,
                obj=pt
            )
            logging.info("Saved power transformer")

            num_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

            preprocessing_obj = self.get_preprocessor_object(num_cols, cat_cols)

            train_arr = preprocessing_obj.fit_transform(train_df)
            test_arr = preprocessing_obj.transform(test_df)
            logging.info(f"Transformation complete. Train shape: {train_arr.shape}")

            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Saved preprocessing object")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)