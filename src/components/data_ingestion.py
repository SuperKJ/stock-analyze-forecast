from dataclasses import dataclass
import pandas as pd
import os
from datetime import datetime, timedelta
import yfinance as yf
from src.exception import CustomException
from src.logger import logging
import sys

@dataclass
class DataIngestionConfig:
    nse_data_path = os.path.join(os.getcwd(),'data',r'NIFTY50_23April2025\MW-NIFTY-50-23-Apr-2025.csv')


class DataIngestor:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def get_stock_symbols(self):
        try:
            path = self.ingestion_config.nse_data_path
            data_NIFTY = pd.read_csv(path)
            logging.info("NIFTY 50 Data acquired")

            data_NIFTY.columns = data_NIFTY.columns.str.replace('\n', '')
            data_NIFTY.columns = data_NIFTY.columns.str.strip()
            symbols = data_NIFTY['SYMBOL'].unique().tolist()
            logging.info("NIFTY Column Transoformation Done")

            return symbols
        
        except Exception as e:
            logging.error("Not Able to Fetch NIFTY 50 Stock Symbols data")
            raise CustomException(e, sys)
        
    def get_stock_data(self, symbol : str, period = '1y', interval = '1d ', start=(datetime.now() - timedelta(days=364*2)).strftime('%Y-%m-%d'), end = datetime.now().strftime("%Y-%m-%d") ):
        """
        Fetch stock data from Yahoo Finance using yfinance library.
        Parameters:
        symbol (str): Stock symbol to fetch data for.
        """

        try:
            logging.info("Entered get_stock_data function")

            symbol = symbol + ".NS"
            stock = yf.Ticker(symbol)
            df = pd.DataFrame(stock.history(period=period, interval=interval, start=start, end=end))
            logging.info("Stock data acquired from yahoo finance")

            return df, stock
        
        except Exception as e:
            logging.error("Error in getting stock data from yahoo finance",e)
            raise CustomException(e, sys)
            