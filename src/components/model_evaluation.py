import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelEvaluationConfig:
    report_file_path: str = os.path.join("reports", "evaluation_report.csv")


class ModelEvaluation:
    def __init__(self):
        self.evaluation_config = ModelEvaluationConfig()


    def compute_metrics(self, X, labels) -> dict:
        """Compute multiple clustering quality metrics, not just silhouette"""       
        try:
            metrics = {
                "silhouette_score": silhouette_score(X, labels),
                "davies_bouldin_score": davies_bouldin_score(X, labels),
                "calinski_harabasz_score": calinski_harabasz_score(X, labels),
                "n_clusters": len(np.unique(labels)),
                "cluster_sizes": np.bincount(labels).tolist()
            }

            return metrics

        except Exception as e:
            raise CustomException(e, sys)


    def profile_clusters(self, original_df: pd.DataFrame, labels:np.ndarray) -> pd.DataFrame: 
        """Attach cluster labels back to the ORIGINAL (pre-scaling) data,
        so cluster averages are in real, readable units — not scaled numbers."""
        try:
            df = original_df.copy()
            df['Cluster'] = labels

            numeric_cols = df.select_dtypes(include=['int64','float64']).columns
            profile = df.groupby('Cluster')[numeric_cols].mean().round(2)
            return profile

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_model_evaluation(self, X_test, test_labels, original_test_df):
        try:
            logging.info("Starting model evaluation")

            metrics = self.compute_metrics(X_test, test_labels)
            logging.info(f"Evaluation metrics: {metrics}")

            profile = self.profile_clusters(original_test_df, test_labels)
            logging.info(f"Cluster profile:\n{profile}")

            os.makedirs(os.path.dirname(self.evaluation_config.report_file_path), exist_ok=True)
            profile.to_csv(self.evaluation_config.report_file_path)
            logging.info(f"Saved evaluation report to {self.evaluation_config.report_file_path}")


            return metrics, profile

        except Exception as e:
            raise CustomException(e, sys)
