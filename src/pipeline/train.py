import sys
from src.exception import CustomException
from src.components.model_trainer import ModelTrainer
from src.components.data_preparation import DataPrepare
from src.logger import logging


class Pipeline:

    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.data_prepare = DataPrepare()
        
    def lr_training_pipeline(self,symbol):
        try:
            logging.info("Entered the training pipeline for Linear Regression")
            data_lr = self.data_prepare.prepare_data_for_model(symbol)

            logging.info("Data prepared for Linear Regression model")
            X_test, y_pred, y_test, rmse, mae, r2 = self.model_trainer.train_lr(data_lr)

            logging.info("Model trained and predictions made")

            return X_test, y_pred, y_test, rmse, mae, r2
        except Exception as e:
            logging.error("Error in training pipeline", e)
            raise CustomException(e,sys)
        
    def xgb_training_pipeline(self,symbol):
        try:
            logging.info("Entered the training pipeline for XGBoost")
            data_xgb = self.data_prepare.prepare_data_for_model(symbol)

            logging.info("Data prepared for XGBoost model")
            X_test, y_pred, y_test, rmse, mae, r2 = self.model_trainer.train_xgb(data_xgb)

            logging.info("Model trained and predictions made")

            return X_test, y_pred, y_test, rmse, mae, r2
        except Exception as e:
            logging.error("Error in training pipeline", e)
            raise CustomException(e,sys)
        
    def elastic_training_pipeline(self,symbol):
        try:
            logging.info("Entered the training pipeline for ElasticNet")
            data_elastic = self.data_prepare.prepare_data_for_model(symbol)

            logging.info("Data prepared for ElasticNet model")
            X_test, y_pred, y_test, rmse, mae, r2 = self.model_trainer.train_elastic(data_elastic)

            logging.info("Model trained and predictions made")

            return X_test, y_pred, y_test, rmse, mae, r2
        except Exception as e:
            logging.error("Error in training pipeline", e)
            raise CustomException(e,sys)
        
    def lstm_training_pipeline(self,symbol):
        try:
            logging.info("Entered the training pipeline for LSTM")
            data_lstm = self.data_prepare.prepare_data_for_model(symbol)

            logging.info("Data prepared for LSTM model")
            X_test, y_pred, y_test, rmse, mae, r2 = self.model_trainer.train_lstm(data_lstm)

            logging.info("Model trained and predictions made")

            return X_test, y_pred, y_test, rmse, mae, r2
        except Exception as e:
            logging.error("Error in training pipeline", e)
            raise CustomException(e,sys)

