import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
import sys
import json
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from datetime import date

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from ta.trend import MACD
from ta.momentum import  RSIIndicator
from xgboost import XGBRegressor
import yfinance as yf
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestor
from src.components.data_preparation import DataPrepare
from src.exception import CustomException
from src.logger import logging
from src.utils import Utils
from sklearn.linear_model import ElasticNet, LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


@dataclass
class ModelTrainerConfig:
    lr_model_trained_path = os.path.join('artifacts', 'models','LinearRegression','lr_model.pkl')
    xgb_model_trained_path = os.path.join('artifacts','models','XGB','xgb_model.pkl')
    elasticnet_model_trained_path = os.path.join('artifacts','models','ElasticNet','en_model.pkl')
    prophet_model_trained_path = os.path.join('artifacts','models','ElasticNet','prophet_model.pkl')
    lstm_model_trained_path = os.path.join('artifacts','models','lstm','lstm_model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.utils = Utils()
        self.data_prepare = DataPrepare()

    def train_lr(self,df):
        """
        Trains a Linear Regression Model for prediction and forecasting
        Parameters:
        df(DataFrame) : 
        """
        try:
            logging.info("Entered Linear Regression Trainer")
            train_data, test_data = self.utils.time_train_test_split(df, test_size=0.2)
            X_train, X_test, y_train, y_test = self.utils.prepare_x_y(train_data, test_data) 

            model_lr = LinearRegression()
            model_lr.fit(X_train, y_train)

            y_pred = model_lr.predict(X_test)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)   
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            logging.info("r2: ", r2, "mae ", mae, "rmse: ", rmse)
            print(r2, mae, rmse)

            self.utils.save_object(self.model_trainer_config.lr_model_trained_path,model_lr)
            logging.info("Model saved successfully (LR) -> ",self.model_trainer_config.lr_model_trained_path)

            return X_test, y_pred, y_test, rmse, mae, r2
        except Exception as e:
            logging.error("Error in training LR Model")
            raise CustomException(e,sys)
    
    def train_elastic(self, df):
        try:
            train_data, test_data = self.utils.time_train_test_split(df, test_size=0.2)
            X_train, X_test, y_train, y_test = self.utils.prepare_x_y(train_data, test_data) 

            elastic = ElasticNet()

            param_grid = {
                'alpha': [0.01, 0.1, 1, 10],           # regularization strength
                'l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0], # 0 = Ridge, 1 = Lasso, in between = ElasticNet
                'fit_intercept': [True, False],
                'max_iter': [1000, 5000]               # ensure convergence
            }

            # GridSearchCV setup
            grid_search = GridSearchCV(
                estimator=elastic,
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=5,
                verbose=1,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)

            print("Best Parameters:", grid_search.best_params_)
            best_model = grid_search.best_estimator_

            y_pred_elastic = best_model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
            r2 = r2_score(y_test, y_pred_elastic)
            mae = mean_absolute_error(y_test, y_pred_elastic) 
            print("ElasticNet RMSE:", rmse)

            self.utils.save_object(self.model_trainer_config.elasticnet_model_trained_path,best_model)
            logging.info("Model saved successfully (ElasticNet) -> ",self.model_trainer_config.elasticnet_model_trained_path)

            return X_test, y_pred_elastic, y_test, rmse, mae, r2
        except Exception as e:
            logging.error("Error in training Elastic Model")
            raise CustomException(e,sys)
        
    def train_xgb(self, df):
        try:
            train_data, test_data = self.utils.time_train_test_split(df, test_size=0.2)
            X_train, X_test, y_train, y_test = self.utils.prepare_x_y(train_data, test_data) 

            xgb = XGBRegressor(eval_metric = "rmse", random_state=42)

            param_grid = {
                'n_estimators': [100, 200, 300],              # Number of trees
                'learning_rate': [0.01, 0.05, 0.1],           # Learning rate
                'max_depth': [3, 5, 7, 10],                    # Maximum depth of each tree
                'min_child_weight': [1, 3, 5],                  # Minimum sum of instance weight (leaf node)
                'subsample': [0.7, 0.8, 0.9],                  # Fraction of samples used for training
                'colsample_bytree': [0.7, 0.8, 0.9],          # Fraction of features used for each tree
                'gamma': [0, 0.1, 0.2],                       # Regularization parameter
                'reg_alpha': [0, 0.1, 0.2],                   # L1 regularization
                'reg_lambda': [0, 0.1, 0.2]                   # L2 regularization
            }

            grid_xg_model = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=param_grid,
                scoring='neg_root_mean_squared_error',
                verbose=1,
                n_jobs=None,
                cv=3  # Use all available cores
            )

            grid_xg_model.fit(X_train, y_train)
            print("Best_Parmas: ", grid_xg_model.best_params_)
            print("Best_Score: ", grid_xg_model.best_score_)
            print("Best Estimator", grid_xg_model.best_estimator_)

            xgb_best = grid_xg_model.best_estimator_
            ypred_xgb= xgb_best.predict(X_test)
            
            r2 = r2_score(y_test, ypred_xgb)
            mae = mean_absolute_error(y_test, ypred_xgb)   
            rmse = np.sqrt(mean_squared_error(y_test, ypred_xgb))

            self.utils.save_object(self.model_trainer_config.xgb_model_trained_path,xgb_best)
            logging.info("Model saved successfully (XGB) -> ",self.model_trainer_config.xgb_model_trained_path)

            return X_test, ypred_xgb, y_test, rmse, mae, r2
        except Exception as e:
            logging.error("Error in training XGB Model")
            raise CustomException(e,sys)
        
    def create_lstm_model(self,input_shape):
        """
        Create and compile an LSTM model.
        Parameters:
        input_shape (tuple): Shape of the input data.
        Returns:
        Model: Compiled LSTM model.
        """

        try:
            logging.info("Creating LSTM Model")
            model = Sequential()

            # 1st Layer: Bidirectional LSTM
            model.add(Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.3))

            # 2nd Layer: LSTM
            model.add(LSTM(32, return_sequences=False))
            model.add(Dropout(0.3))

            # Output Layer
            model.add(Dense(1))  # Predict single value (like next Close Price)

            model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

            logging.info("LSTM Model Created!!")
            
            return model
        except Exception as e:
            logging("There has been an error in creating the LSTM Model")
            raise CustomException(e,sys)
    
    def train_lstm_model(self,model, X, y, epochs=50, batch_size=32):
        try: 
            logging.info("Training LSTM Model")
            callback_es = tf.keras.callbacks.EarlyStopping(monitor='mse', patience=15, restore_best_weights=True, mode='min')
            callback_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='mse', factor=0.2, patience=15, min_lr=0.00001)
            history = model.fit(X, y, epochs=epochs, batch_size=batch_size,callbacks=[callback_es, callback_lr], verbose=1)
            logging.info("LSTM Model Trained")
            return history
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def train_lstm(self,df):
        try:
            logging.info("Training LSTM Model")

            # Split into train and test sets
            train_data, test_data = self.utils.time_train_test_split(df, test_size=0.2)
            X_train, X_test, y_train, y_test = self.utils.prepare_x_y(train_data, test_data)

            # Define features and target
            features = ['sma_10', 'MACD', 'MACD_Diff', 'RSI', 'Close_prev', 'dayofweek', 'day', 'month', 'year']
            pred = ['Close']

            # Scale the features and target
            X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x, scaler_y = self.utils.scaled_features(
                features, pred, X_train, X_test, y_train, y_test
            )

            # Create sequences for LSTM
            X_train_seq, y_train_seq = self.data_prepare.create_seq_for_lstm(X_train_scaled, y_train_scaled)
            X_test_seq, y_test_seq = self.data_prepare.create_seq_for_lstm(X_test_scaled, y_test_scaled)

            # Input shape for the LSTM model
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

            # Create and train LSTM model
            model_lstm = self.create_lstm_model(input_shape)
            model_lstm_history = self.train_lstm_model(model_lstm, X_train_seq, y_train_seq, epochs=200, batch_size=32)

            # Predict on test sequences
            y_pred_lstm_scaled = model_lstm.predict(X_test_seq)

            logging.info("Predictions on LSTM Test set done")

            # Inverse transform predictions and actuals
            y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled)
            y_pred_lstm = y_pred_lstm.reshape(-1,)

            y_lstm = scaler_y.inverse_transform(y_test_seq)
            y_lstm = np.squeeze(y_lstm)

            # Evaluation metrics
            r2 = r2_score(y_lstm, y_pred_lstm)
            mae = mean_absolute_error(y_lstm, y_pred_lstm)
            mse = mean_squared_error(y_lstm, y_pred_lstm)
            rmse = np.sqrt(mse)

            print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

            # Save trained model
            self.utils.save_object(self.model_trainer_config.lstm_model_trained_path, model_lstm)
            logging.info(f"Model saved successfully (LSTM) -> {self.model_trainer_config.lstm_model_trained_path}")

            return X_test, y_pred_lstm, y_lstm, rmse, mae, r2

        except Exception as e:
            logging.error("Error in training LSTM Model")
            raise CustomException(e, sys)






    