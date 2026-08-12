import sys
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation

from src.exception import CustomException
from src.logger import logging


class TrainPipeline:
    def run_pipeline(self):
        try:
            logging.info("=== Training Pipeline Started ===")

            # Step 1: Ingest
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()


            # Step 2: Validate
            validation = DataValidation()
            is_valid = validation.initiate_data_validation(train_path, test_path)
            if not is_valid:
                raise CustomException("Data validaiton failde. Stopping pipeline.", sys)


            # Step 3: Transform
            transformation = DataTransformation()
            train_arr, test_arr, _ = transformation.initiate_data_transformation(train_path, test_path)


            # Step 4: Train
            trainer = ModelTrainer()
            train_labels, test_labels, score = trainer.initiate_model_training(train_arr, test_arr)


            # Step 5: Evaluate
            original_test_df = pd.read_csv(test_path)
            evaluation = ModelEvaluation()
            metrics, profile = evaluation.initiate_model_evaluation(test_arr, test_labels, original_test_df)

            logging.info(f"==== Training Pipeline Complete. Silhouette: {score:.4f} ===")

            return score, metrics, profile

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainPipeline()
    final_score, metrics, profile = pipeline.run_pipeline()
    print(f"Pipeline finished. Final silhouette score: {final_score:.4f}")
    print(f"Full metrics: {metrics}")
    print(f"\nCluster profile:\n{profile}")        