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
from dataclasses import dataclass
from src.components.data_ingestion import DataIngestor
from src.exception import CustomException
from src.logger import logging

class DataPrepare:
    def __init__(self):
        self.data_ingestor= DataIngestor()
    
    def calculate_mathematical_term(self,df):
        """
        Calculate SMA, MACDa dn RSI for stock data.
        Parameters:
        df (DataFrame): DataFrame containing stock data.
        Returns:
        DataFrame: DataFrame with SMA(10 days, 50 days and 200days), MACD and RSI columns added.
        """
        try:
            logging.info("entered calculate_mathematical_term")
            #SMA(50 days and 200days)
            sma_50 = df['Close'].rolling(window=50).mean()
            sma_200 = df['Close'].rolling(window=200).mean()
            sma_10 = df['Close'].rolling(window=10).mean()
            df['sma_50'] = sma_50
            df['sma_200'] = sma_200
            df['sma_10'] = sma_10

            #MACD
            macd = MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['Signal_Line'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()

            #RSI
            rsi = RSIIndicator(close=df['Close'], window=14)
            df['RSI'] = rsi.rsi()

            logging.info("Calculated the required mathematical terms and returning a data frame")

            return df
        
        except Exception as e:
            logging.error("Error in Calculating Mathematical Terms",e)
            raise CustomException(e,sys)
    
    def prepare_data_for_model(self,symbol):
        """
        Prepare data for training the ML models.
        Parameters: 
        symbol(string) : the symbol of stock 
        Returns:
        DataFrame with required columns for training
        """

        try:

            logging.info("Entered prepare_data_for_model")
            data, stock= self.data_ingestor.get_stock_data(symbol,start=(datetime.now() - timedelta(days=364*3)).strftime('%Y-%m-%d'))

            data = self.calculate_mathematical_term(data)

            data.drop([ 'sma_50', 'sma_200', 'Dividends', 'Stock Splits','Open', 'High', 'Low', 'Signal_Line', 'Volume'], axis=1, errors='ignore',inplace=True)

            #(data.columns)

            data.reset_index( inplace=True)

            data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
            #(data.columns)
            
            data['Close_prev'] = data['Close'].shift(1)
            
            data.dropna(axis=0, inplace=True)
        
            data['dayofweek'] = data['Date'].dt.dayofweek          # 0=Monday
            data['day'] = data['Date'].dt.day
            data['month'] = data['Date'].dt.month
            data['year'] = data['Date'].dt.year
            
            data.set_index('Date', inplace=True)

            #("data prepared for ML models",data.columns)
            
            logging.info("data for machine learning models is prepared")
            return data
        
        except Exception as e:
            logging.error("Error in preparing data for ML Models: ")
            raise CustomException(e,sys)
    
    
    def create_seq_for_lstm(self,X,y, timestep=10):
        """
        Create sequences of input features and corresponding target values for LSTM models.
        Parameters: X (features), y (targets), timestep (sequence length).
        Returns: 3D array of sequences (Xs) and 1D array of targets (ys).
        """
        X = X
        y = y
        Xs, ys = [], []
        for i in range(timestep, X.shape[0]):
            Xs.append(X[i - timestep:i])  
            ys.append(y[i])

        return np.array(Xs), np.array(ys)
