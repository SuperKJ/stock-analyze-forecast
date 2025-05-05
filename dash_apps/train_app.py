from datetime import date
import dash
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import State, html,Input, Output, dcc
from flask import app
from src.components.data_ingestion import DataIngestor
from src.components.data_preparation import DataPrepare
from datetime import datetime, timedelta
import dash_apps.dash_utils as du 
from src.pipeline.train import Pipeline
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict import PredictionPipeline


def create_train_dash(server):
    dash_app = dash.Dash(
        __name__, server=server, url_base_pathname='/train/',
        suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG],  meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0",
        }
    ],
    )

    data_ingestor = DataIngestor()
    data_prepare = DataPrepare()
    model_trainer = ModelTrainer()
    train_pipeline = Pipeline()
    predict_pipeline = PredictionPipeline()
    symbols_list = data_ingestor.get_stock_symbols()
    symbols_list.remove("NIFTY 50")
    models_list = ['Linear Regression','XG Boost', 'ElasticNet', 'LSTM']
    days = [5,10,25]

    dash_app.layout = html.Div([
            # Navbar
            dbc.Nav([
                html.Div([
                    html.A("Stock Dashboard", className="navbar-brand", href="/"),
                    html.Ul([
                        html.Li(html.A("Analyze", href="/analyze/", className="nav-link"), className="nav-item"),
                        html.Li(html.A("Train & Forecast", href="/train/", className="nav-link"), className="nav-item"),
                    ], className="navbar-nav me-auto")
                ], className="container-fluid")
            ], className="navbar navbar-expand-lg navbar-dark bg-dark"),
            
            # Main Page Content

        html.Br(),
        dbc.Container([
            html.H2("Machine Learning Models Training and Forecasting Dashboard", className="text-warning"),
            html.Br(),

            

            dbc.Row([
                dbc.Col([
                    dbc.Select(
                        id='symbol',
                        options=[{'label': sym, 'value': sym} for sym in symbols_list],
                        placeholder="Select A Stock Symbol to Display the Dashboard..."
                    )
                ]),
                dbc.Col([
                    dbc.Select(
                        id='model-input',
                        options=[{'label': model, 'value': model} for model in models_list],
                        placeholder="Select A Model..."
                    )
                ]),
                dbc.Col([
                    dbc.Select(
                        id='days-input',
                        options=[{'label': day, 'value': day} for day in days],
                        placeholder="Select how many days to forecast..."
                    )
                ]),
                dbc.Col([
                    dbc.Button('Train', id='submit-val', n_clicks=0)
                ]),

                html.Br(),
                html.Br(),

                html.Hr(),

                html.Br(),

                dbc.Container(
                    [
                        dbc.Spinner(html.Div(id="accuracy-row")),
                        html.Br(),
                        

                    ],
                ),
            ]),


        ])

    ])

    @dash_app.callback(
        Output('accuracy-row','children'),
        Input("submit-val","n_clicks"),
        State("symbol","value"),
        State('model-input', 'value'),
        State('days-input', 'value'),
        prevent_initial_call=True
    )
    def train_model(n_clicks,symbol,model,n_days):

        #(model)
        #(symbol)

        if model == "Linear Regression":
            #(model)
            #(symbol)
            X_test, y_pred, y_test, rmse, mae, r2 = train_pipeline.lr_training_pipeline(symbol)
            forecast_data = predict_pipeline. make_prediction_ml_model(model,symbol,n_days)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x= X_test.index, y=y_test, mode='lines', name='Test Data Actuals'))
            fig.add_trace(go.Scatter(x= X_test.index, y=y_pred, mode='lines', name='Predictions'))

            fig.update_layout(
                title="Linear Regression Predictions vs Actual",
                xaxis_title="Date",
                yaxis_title="Close Price",
                xaxis_rangeslider_visible=True,
                xaxis_rangeslider_thickness=0.05,
                xaxis_rangeslider_bgcolor='rgba(0, 0, 0, 0.1)',
                xaxis_rangeslider_bordercolor='rgba(0, 0, 0, 0.1)',   
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)')
            
            #("Model Training COmplemeted")
            #("Forecasting COmplemeted")

            forecast_data = forecast_data.reset_index().loc[:, ['index', 'Close']].rename(columns={'index': 'Date'})
            forecast_data['Date'] = pd.to_datetime(forecast_data['Date']).dt.date
            #(forecast_data)
            
            return html.Div([du.plot_train_pred_forecast(rmse, mae,r2,fig, forecast_data, symbol)])

        
        elif model == "XG Boost":
            #(model)
            #(symbol)
            X_test, y_pred, y_test, rmse, mae, r2 = train_pipeline.xgb_training_pipeline(symbol)

            #("Model Training COmplemeted")
            forecast_data = predict_pipeline. make_prediction_ml_model(model,symbol,int(n_days))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x= X_test.index, y=y_test, mode='lines', name='Test Data Actuals'))
            fig.add_trace(go.Scatter(x= X_test.index, y=y_pred, mode='lines', name='Predictions'))

            fig.update_layout(
                title="XGBoost Predictions vs Actual",
                xaxis_title="Date",
                yaxis_title="Close Price",
                xaxis_rangeslider_visible=True,
                xaxis_rangeslider_thickness=0.05,
                xaxis_rangeslider_bgcolor='rgba(0, 0, 0, 0.1)',
                xaxis_rangeslider_bordercolor='rgba(0, 0, 0, 0.1)',   
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)')
            
            #(forecast_data)
            
            #('forecast_data.shape', forecast_data.shape)
            #('forecast_Data.colums', forecast_data.columns)            
            forecast_data = forecast_data.reset_index().loc[:, ['index', 'Close']].rename(columns={'index': 'Date'})
            forecast_data['Date'] = pd.to_datetime(forecast_data['Date']).dt.date
            #(forecast_data)
            
            return html.Div([du.plot_train_pred_forecast(rmse, mae,r2,fig, forecast_data, symbol)])
        
        elif model == "ElasticNet":
            #(model)
            #(symbol)
            print(model)
            X_test, y_pred, y_test, rmse, mae, r2 = train_pipeline.elastic_training_pipeline(symbol)

            #("Model Training COmplemeted")
            forecast_data = predict_pipeline. make_prediction_ml_model(model,symbol,int(n_days))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x= X_test.index, y=y_test, mode='lines', name='Test Data Actuals'))
            fig.add_trace(go.Scatter(x= X_test.index, y=y_pred, mode='lines', name='Predictions'))

            fig.update_layout(
                title="XGBoost Predictions vs Actual",
                xaxis_title="Date",
                yaxis_title="Close Price",
                xaxis_rangeslider_visible=True,
                xaxis_rangeslider_thickness=0.05,
                xaxis_rangeslider_bgcolor='rgba(0, 0, 0, 0.1)',
                xaxis_rangeslider_bordercolor='rgba(0, 0, 0, 0.1)',   
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)')
            
            #("ElasticNet Model Training COmplemeted")
            
            forecast_data = forecast_data.reset_index().loc[:, ['index', 'Close']].rename(columns={'index': 'Date'})
            forecast_data['Date'] = pd.to_datetime(forecast_data['Date']).dt.date
            #(forecast_data)
            
            return html.Div([du.plot_train_pred_forecast(rmse, mae,r2,fig, forecast_data, symbol)])
        
        elif model == "LSTM":
            #(model)
            #(symbol)
            X_test, y_pred, y_test, rmse, mae, r2 = train_pipeline.lstm_training_pipeline(symbol)

            #("Model Training COmplemeted")
            forecast_data = predict_pipeline.make_prediction_lstm_model(symbol,int(n_days))

            fig = go.Figure()
            fig.add_trace(go.Scatter(x= X_test.index, y=y_test, mode='lines', name='Test Data Actuals'))
            fig.add_trace(go.Scatter(x= X_test.index, y=y_pred, mode='lines', name='Predictions'))

            fig.update_layout(
                title="LSTM Predictions vs Actual",
                xaxis_title="Date",
                yaxis_title="Close Price",
                xaxis_rangeslider_visible=True,
                xaxis_rangeslider_thickness=0.05,
                xaxis_rangeslider_bgcolor='rgba(0, 0, 0, 0.1)',
                xaxis_rangeslider_bordercolor='rgba(0, 0, 0, 0.1)',   
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)')
            
            #("ElasticNet Model Training COmplemeted")
            
            # forecast_data = forecast_data.reset_index().loc[:, ['index', 'Close']].rename(columns={'index': 'Date'})
            # forecast_data['Date'] = pd.to_datetime(forecast_data['Date']).dt.date
            #(forecast_data)
            
            return html.Div([du.plot_train_pred_forecast(rmse, mae,r2,fig, forecast_data, symbol)])

            
    
        
