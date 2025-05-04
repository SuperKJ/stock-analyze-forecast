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
from ta.trend import MACD
from ta.momentum import  RSIIndicator
import yfinance as yf
from src.components.data_ingestion import DataIngestor
from src.logger import logging
from sklearn.preprocessing import MinMaxScaler
import dill
import pickle
from src.exception import CustomException


class Utils:
    def  time_train_test_split(self, data, test_size=0.2):
        """
        Split the data into training and testing sets.

        Returns:
        DataFrame: Training set.
        DataFrame: Testing set.
        """
        
        train_size = int(len(data) * (1 - test_size))
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        return train_data, test_data
    
    def prepare_x_y(self, train_data, test_data):
        """
        Prepare X and y for training and testing.
        Parameters:
        train_data (DataFrame): Training data.
        test_data (DataFrame): Testing data.
        Returns:
        DataFrame: X_train.
        DataFrame: X_test.
        Series: y_train.
        Series: y_test.
        """
        X_train = train_data.drop(['Close'], axis=1)
        X_test = test_data.drop(['Close'], axis=1)
        y_train = train_data['Close']
        y_test = test_data['Close']
        
        return X_train, X_test, y_train, y_test
    
    def scaled_features(self, features,pred,X_train, X_test, y_train, y_test):
        """
        Scale features using MinMaxScaler.
        Parameters:
        features (list): List of feature names to scale.
        data (DataFrame): DataFrame containing the data.
        Returns:
        DataFrame: Scaled DataFrame.
        """
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_x.fit_transform(X_train[features])
        X_test_scaled = scaler_x.fit_transform(X_test[features])
        y_train_scaled = scaler_y.fit_transform(y_train.to_frame())
        y_test_scaled = scaler_y.fit_transform(y_test.to_frame())
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x, scaler_y
    
    
    def save_object(self, file_path, obj):
        """
        Save the object to a file using pickle.
        """
        try:
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'wb') as file:
                dill.dump(obj, file)
            logging.info(f"Object saved to {file_path}")
        except Exception as e:
            raise CustomException(e, sys) from e
    

    
    def load_object(self, file_path):
        try:
            with open(file_path, "rb") as file_obj:
                return pickle.load(file_obj)

        except Exception as e:
            raise CustomException(e, sys)
        
    def calculate_macd(self,close_prices, slow=26, fast=12):
        exp1 = close_prices.ewm(span=fast, adjust=False).mean()
        exp2 = close_prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_diff = macd - signal
        return macd.iloc[-1], signal.iloc[-1], macd_diff.iloc[-1]

    def calculate_rsi(self,close_prices, window=14):
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_sma(self,close_prices, window):
        return close_prices.rolling(window=window).mean().iloc[-1]