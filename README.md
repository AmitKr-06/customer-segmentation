# Customer Segmentation using Unsupervised Learning

Segments retail customers into distinct behavioral groups using K-Means clustering,
so marketing teams can target each group differently instead of treating all
customers the same. Deployed as a REST API for real-time segment prediction.

## Problem Statement

The business has customer data but no existing "segment" labels — there's no ground
truth to predict, only patterns to discover. This project uses unsupervised learning
to find natural groupings in customer demographics, spending, and engagement behavior,
and exposes the trained model through an API for integration into other systems.

## Dataset

Customer Personality Analysis dataset — 2,240 rows, 29 columns, covering demographics
(income, education, family), spending across product categories, and marketing
campaign response history.

## Project Structure

```
customer-segmentation/
├── data/
│   ├── raw/                       # Original CSV
│   └── processed/                 # Train/test splits
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── components/
│   │   ├── data_ingestion.py      # Loads raw data, splits train/test
│   │   ├── data_validation.py     # Checks data quality before processing
│   │   ├── data_transformation.py # Feature engineering, encoding, scaling, PCA
│   │   ├── model_trainer.py       # Trains and selects the K-Means model
│   │   └── model_evaluation.py    # Computes metrics, profiles clusters
│   ├── pipeline/
│   │   ├── train_pipeline.py      # Runs all components end-to-end
│   │   └── predict_pipeline.py    # Loads saved model, predicts for new data
│   ├── exception.py                # Custom error handling with file/line detail
│   ├── logger.py                   # Timestamped run logging
│   └── utils.py                    # save_object / load_object helpers
├── models/                         # Saved preprocessor, transformer, model (generated)
├── logs/                           # Run logs (generated)
├── reports/                        # Evaluation report (generated)
├── app.py                          # FastAPI app serving predictions
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone <your-repo-url>
cd customer-segmentation
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the Training Pipeline

```bash
python -m src.pipeline.train_pipeline
```

This runs ingestion → validation → feature engineering → scaling/PCA →
K-Means training → evaluation, and saves all model artifacts to `models/`.

## Running the API Locally

```bash
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` for an interactive API testing interface.

**Example request to `/predict`:**
```json
{
  "Year_Birth": 1985,
  "Education": "Graduation",
  "Marital_Status": "Married",
  "Income": 58000,
  "Kidhome": 1,
  "Teenhome": 0,
  "Dt_Customer": "15-06-2013",
  "Recency": 30,
  "MntWines": 500,
  "MntFruits": 20,
  "MntMeatProducts": 150,
  "MntFishProducts": 30,
  "MntSweetProducts": 15,
  "MntGoldProds": 40,
  "NumDealsPurchases": 2,
  "NumWebPurchases": 5,
  "NumCatalogPurchases": 1,
  "NumStorePurchases": 4,
  "NumWebVisitsMonth": 6,
  "AcceptedCmp1": 0,
  "AcceptedCmp2": 0,
  "AcceptedCmp3": 0,
  "AcceptedCmp4": 0,
  "AcceptedCmp5": 0,
  "Complain": 0,
  "Response": 0
}
```

**Response:**
```json
{ "predicted_segment": 1 }
```

## Methodology

1. **EDA** — univariate/bivariate analysis, correlation checks, distribution checks
2. **Feature Engineering** — created `Age`, `TotalSpend`, `TotalChildren`,
   `CustomerTenureDays`, `DealAffinity`, `TotalCampaignsAccepted`; fixed skew
   in `Income`, `TotalSpend`, `TotalCampaignsAccepted` with a Yeo-Johnson power transform
3. **Preprocessing** — one-hot encoding, standard scaling, PCA (95% variance retained)
4. **Model Selection** — compared K-Means, Agglomerative, GMM, and DBSCAN

## Why K-Means, Not DBSCAN

DBSCAN scored a higher raw silhouette (~0.81), but only by classifying 99%+ of
customers as "noise" and clustering a handful of outliers. K-Means (silhouette ≈ 0.26)
assigns every customer to one of 2 balanced, usable segments, making it the practical
choice for real segmentation — a model that refuses to classify almost the entire
dataset isn't a usable business tool, regardless of its score.

## Results

- **Final model:** K-Means, k=2
- **Silhouette score:** 0.2426 (test set)
- **Davies-Bouldin score:** 1.77
- **Calinski-Harabasz score:** 126.5
- **Segment sizes (test set):** 190 / 258 customers — balanced, no data discarded

**Segment profile:**

| | Income | Kidhome | Teenhome |
|---|---|---|---|
| Segment 0 | $71,522 | 0.09 | 0.42 |
| Segment 1 | $39,760 | 0.62 | 0.59 |

**Segment 0** — higher-income customers, fewer children at home.
**Segment 1** — lower-income customers, more children at home.

## Tech Stack

Python, pandas, scikit-learn, FastAPI, Uvicorn, seaborn, matplotlib

## Deployment

Deployed to AWS as a containerized FastAPI service. See deployment notes below
*(update this section once AWS deployment is complete)*.