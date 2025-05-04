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
from src.components.data_preparation import DataPrepare
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from flask import Flask, render_template
import dash_apps.stock_eda as eda
import dash_apps.train_app as train


app = Flask(__name__)

eda_app = eda.create_eda_dash(app)
train_app = train.create_train_dash(app)

@app.route('/')
def home_page():
    return render_template("index.html")



if __name__ == "__main__":
    app.run(debug=True)

    # data_ingestor = DataIngestor()
    # symbols = data_ingestor.get_stock_symbols()

    # symbol = symbols[1]

    # data_for_plotting = data_ingestor.get_stock_data(symbol)

    # data_prepare = DataPrepare()
    # data_for_training = data_prepare.prepare_data_for_model(symbol)

    # model_trainer = ModelTrainer()
    # y_pred, y_test, rmse = model_trainer.train_lstm(data_for_training)
    



    
