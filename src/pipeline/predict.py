import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from src.utils import Utils
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.components.data_preparation import DataPrepare
from src.components.data_ingestion import DataIngestor


class PredictionPipeline:
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.data_prepare = DataPrepare()
        self.data_ingestor = DataIngestor()
        self.utils = Utils()
        self.model_config = ModelTrainerConfig()

    def make_prediction_ml_model(self,model,symbol, n_days = 10):
        n_days = int(n_days)

        data, stock = self.data_ingestor.get_stock_data(symbol)
        data = self.data_prepare.calculate_mathematical_term(data)
        data = self.data_prepare.prepare_data_for_model(symbol)

        print('Last Close Price: ', data['Close'].iloc[-1])
        print('Last Date: ', data.index[-1])

        predictions = pd.DataFrame(columns=['sma_10', 'MACD', 'MACD_Diff', 'RSI', 'Close_prev', 'dayofweek', 'day', 'month', 'year', 'Close'])

        data_for_predictions = data.copy()

        for i in range(n_days):
            future = pd.to_datetime((datetime.now() + timedelta(days=i)))
            
            pred_dict = {
                'sma_10': self.utils.calculate_sma(data_for_predictions['Close'], 10),
                'MACD': self.utils.calculate_macd(data_for_predictions['Close'])[0],
                'MACD_Diff': self.utils.calculate_macd(data_for_predictions['Close'])[2],
                'RSI': self.utils.calculate_rsi(data_for_predictions['Close']),  
                'Close_prev': data_for_predictions['Close'].iloc[-1],
                'dayofweek': future.dayofweek, # 0=Monday
                'day' : future.day,
                'month': future.month,
                'year': future.year,     
            }

            df_pred = pd.DataFrame(pred_dict, index=[future])

            if model == "Linear Regression":
                model = self.utils.load_object(self.model_config.lr_model_trained_path)
            elif model == "XG Boost":
                model = self.utils.load_object(self.model_config.xgb_model_trained_path)
            elif model == "ElasticNet":
                model = self.utils.load_object(self.model_config.elasticnet_model_trained_path)
                print("ElasticNet model loaded")

            y_pred = model.predict(df_pred)

            df_pred['Close'] = y_pred[0]

            data_for_predictions = pd.concat([data_for_predictions, df_pred])

            predictions = pd.concat([predictions, df_pred])

        print('Predicted Close Price: ', predictions)

        return predictions
    
    def make_prediction_lstm_model(self,symbol, n_days: int):
        # Prepare the data
        data = self.data_prepare.prepare_data_for_model(symbol)

        print("Prediction data columns:", data.columns)
        print("Data shape:", data.shape)

        # Load the trained LSTM model
        model = self.utils.load_object(self.model_config.lstm_model_trained_path)
        print("LSTM model loaded")

        # Drop target column to get features
        X = data.drop(['Close'], axis=1)
        y = data['Close']

        # Scale the features and target
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_scaled = scaler_x.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))

        # Create sequences
        timestep = 10
        X_train_seq, y_train_seq = self.data_prepare.create_seq_for_lstm(X_scaled, y_scaled)

        # Forecasting setup
        last_sequence = X_train_seq[-1]  # Shape: (timestep, num_features)
        forecasted = []

        for _ in range(n_days):
            input_seq = last_sequence.reshape(1, timestep, -1)  # (1, 10, num_features)

            # Predict scaled Close
            pred_scaled = model.predict(input_seq)
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]
            forecasted.append(pred)

            # Update the 'Close' in the feature vector with the prediction
            new_feature_vector = input_seq[0, -1, :].copy()
            new_feature_vector[0] = pred_scaled[0][0]  # Assuming Close is at index 0

            # Build new sequence for the next prediction
            new_input = np.vstack([last_sequence[1:], new_feature_vector])  # Shape: (10, num_features)
            last_sequence = new_input

        # Build forecast DataFrame
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
        forecast_df = pd.DataFrame({'Date': future_dates, 'Close': forecasted})


        return forecast_df