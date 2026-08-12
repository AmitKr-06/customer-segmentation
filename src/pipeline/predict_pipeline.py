import sys
import json
import pandas as pd

from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        self.preprocessor_path = "models/preprocessor.pkl"
        self.model_path = "models/kmeans_model.pkl"
        self.power_transformer_path = "models/power_transformer.pkl"
        self.reference_date_path = "models/reference_date.json"

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mirrors DataTransformation.engineer_features exactly, for one new row."""
        with open(self.reference_date_path, "r") as f:
            max_date = pd.to_datetime(json.load(f)["max_dt_customer"])

        df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], dayfirst=True)
        df['Age'] = 2025 - df['Year_Birth']
        df['TotalChildren'] = df['Kidhome'] + df['Teenhome']

        mnt_cols = ['MntWines', 'MntFruits', 'MntMeatProducts',
                    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
        df['TotalSpend'] = df[mnt_cols].sum(axis=1)

        purchase_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
        df['TotalPurchases'] = df[purchase_cols].sum(axis=1)
        df['DealAffinity'] = df['NumDealsPurchases'] / (df['TotalPurchases'] + 1)

        df['CustomerTenureDays'] = (max_date - df['Dt_Customer']).dt.days

        campaign_cols = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
        df['TotalCampaignsAccepted'] = df[campaign_cols].sum(axis=1)

        drop_cols = ['Kidhome', 'Teenhome', 'ID', 'Z_CostContact', 'Z_Revenue',
                     'TotalPurchases', 'Year_Birth', 'Dt_Customer']
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        return df

    def predict(self, features: pd.DataFrame) -> int:
        try:
            pt = load_object(self.power_transformer_path)
            preprocessor = load_object(self.preprocessor_path)
            model = load_object(self.model_path)

            features = self.engineer_features(features)

            skewed_cols = ['Income', 'TotalSpend', 'TotalCampaignsAccepted']
            features[skewed_cols] = pt.transform(features[skewed_cols])

            data_transformed = preprocessor.transform(features)
            cluster = model.predict(data_transformed)

            return int(cluster[0])

        except Exception as e:
            raise CustomException(e, sys)


class CustomerData:
    """Wraps raw input into the exact structure the pipeline expects."""

    def __init__(self, Year_Birth, Education, Marital_Status, Income, Kidhome,
                 Teenhome, Dt_Customer, Recency, MntWines, MntFruits,
                 MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds,
                 NumDealsPurchases, NumWebPurchases, NumCatalogPurchases,
                 NumStorePurchases, NumWebVisitsMonth, AcceptedCmp1, AcceptedCmp2,
                 AcceptedCmp3, AcceptedCmp4, AcceptedCmp5, Complain, Response):
        self.data = {
            "Year_Birth": [Year_Birth], "Education": [Education],
            "Marital_Status": [Marital_Status], "Income": [Income],
            "Kidhome": [Kidhome], "Teenhome": [Teenhome],
            "Dt_Customer": [Dt_Customer], "Recency": [Recency],
            "MntWines": [MntWines], "MntFruits": [MntFruits],
            "MntMeatProducts": [MntMeatProducts], "MntFishProducts": [MntFishProducts],
            "MntSweetProducts": [MntSweetProducts], "MntGoldProds": [MntGoldProds],
            "NumDealsPurchases": [NumDealsPurchases], "NumWebPurchases": [NumWebPurchases],
            "NumCatalogPurchases": [NumCatalogPurchases], "NumStorePurchases": [NumStorePurchases],
            "NumWebVisitsMonth": [NumWebVisitsMonth], "AcceptedCmp1": [AcceptedCmp1],
            "AcceptedCmp2": [AcceptedCmp2], "AcceptedCmp3": [AcceptedCmp3],
            "AcceptedCmp4": [AcceptedCmp4], "AcceptedCmp5": [AcceptedCmp5],
            "Complain": [Complain], "Response": [Response],
        }

    def get_data_as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)


if __name__ == "__main__":
    sample = CustomerData(
        Year_Birth=1985, Education="Graduation", Marital_Status="Married",
        Income=58000, Kidhome=1, Teenhome=0, Dt_Customer="15-06-2013",
        Recency=30, MntWines=500, MntFruits=20, MntMeatProducts=150,
        MntFishProducts=30, MntSweetProducts=15, MntGoldProds=40,
        NumDealsPurchases=2, NumWebPurchases=5, NumCatalogPurchases=1,
        NumStorePurchases=4, NumWebVisitsMonth=6, AcceptedCmp1=0,
        AcceptedCmp2=0, AcceptedCmp3=0, AcceptedCmp4=0, AcceptedCmp5=0,
        Complain=0, Response=0
    )

    df = sample.get_data_as_dataframe()
    predictor = PredictPipeline()
    cluster = predictor.predict(df)
    print(f"Predicted segment: {cluster}")
    