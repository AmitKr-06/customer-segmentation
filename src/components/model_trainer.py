import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("models", "kmeans_model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def find_best_k(self, train_arr, k_range=range(2, 11)):
        """Run K-Means across a range of k and return the one with the best silhouette."""
        best_k = None
        best_score = -1

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(train_arr)
            score = silhouette_score(train_arr, labels)
            logging.info(f"k={k}: silhouette={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        return best_k, best_score

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Finding optimal number of clusters")
            best_k, best_score = self.find_best_k(train_arr)
            logging.info(f"Best k: {best_k}, silhouette: {best_score:.4f}")

            final_model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            train_labels = final_model.fit_predict(train_arr)
            test_labels = final_model.predict(test_arr)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=final_model
            )
            logging.info("Saved final KMeans model")

            return train_labels, test_labels, best_score

        except Exception as e:
            raise CustomException(e, sys)


