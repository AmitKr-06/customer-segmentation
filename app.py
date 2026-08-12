from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipeline.predict_pipeline import PredictPipeline, CustomerData
from src.exception import CustomException

app = FastAPI(title="Customer Segmentation API")


class CustomerInput(BaseModel):
    Year_Birth: int
    Education: str
    Marital_Status: str
    Income: float
    Kidhome: int
    Teenhome: int
    Dt_Customer: str
    Recency: int
    MntWines: int
    MntFruits: int
    MntMeatProducts: int
    MntFishProducts: int
    MntSweetProducts: int
    MntGoldProds: int
    NumDealsPurchases: int
    NumWebPurchases: int
    NumCatalogPurchases: int
    NumStorePurchases: int
    NumWebVisitsMonth: int
    AcceptedCmp1: int
    AcceptedCmp2: int
    AcceptedCmp3: int
    AcceptedCmp4: int
    AcceptedCmp5: int
    Complain: int
    Response: int


@app.get("/")
def read_root():
    return {"message": "Customer Segmentation API is running"}


@app.post("/predict")
def predict_segment(customer: CustomerInput):
    try:
        customer_data = CustomerData(**customer.model_dump())
        df = customer_data.get_data_as_dataframe()

        predictor = PredictPipeline()
        cluster = predictor.predict(df)

        return {"predicted_segment": cluster}

    except CustomException as e:
        raise HTTPException(status_code=500, detail=str(e))