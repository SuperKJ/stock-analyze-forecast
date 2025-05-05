from datetime import date
import dash
import dash_bootstrap_components as dbc
from dash import html,Input, Output, dcc
from flask import app
from src.components.data_ingestion import DataIngestor
from src.components.data_preparation import DataPrepare
from datetime import datetime, timedelta
import dash_apps.dash_utils as du 


def create_eda_dash(server):
    dash_app = dash.Dash(
        __name__, server=server, url_base_pathname='/analyze/',
        suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.CYBORG],  meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0",
        }
    ],
    )

    data_ingestor = DataIngestor()
    data_prepare = DataPrepare()
    symbols_list = data_ingestor.get_stock_symbols()

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
            html.H2("Stock Fundamental and Technical Analysis", className="text-warning"),
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
                    dcc.DatePickerRange(
                        id='my-date-picker-range',
                        min_date_allowed=date(2001, 1, 1),
                        max_date_allowed=datetime.now().strftime('%Y-%m-%d'),
                        end_date=datetime.now().strftime('%Y-%m-%d'),
                        start_date=(datetime.now() - timedelta(days=364*2)).strftime('%Y-%m-%d')
                        
                    ),
                ]),
            ]),

            html.Hr(),

            html.Br(),

            dbc.Container(
                [
                    dbc.Spinner(html.Div(id="card-row")),
                    dbc.Spinner(html.Div(id='graphs')),
                    html.Br(),

                ],
            ),
        ],  

        )  
                            
    ])
            
    
                
    @dash_app.callback(
        Output('card-row', 'children'),
        [
            Input('symbol', 'value'),
            Input('my-date-picker-range', 'start_date'),
            Input('my-date-picker-range', 'end_date')
        ],
        prevent_initial_call=True
    )
    def update_graph(symbol, start_date, end_date):

        data, stock = data_ingestor.get_stock_data(symbol = symbol, start = start_date, end = end_date)
        
        
        name = stock.info.get('shortName', 'N/A')
        currPrice = stock.info.get('currentPrice', 'N/A')
        de = stock.info.get('debtToEquity', 'N/A')
        eps_ttm = stock.info.get('epsTrailingTwelveMonths', 'N/A')

        return dbc.Row(
        [
            dbc.Col(du.make_info_card("Company Name", name), width=3),
            dbc.Col(du.make_info_card("Current Price",  "₹ "+str(currPrice)), width=3),
            dbc.Col(du.make_info_card("Debt to Equity Ratio", str(de) + "%"), width=3),
            dbc.Col(du.make_info_card("EPS (Triling 12 Months)", eps_ttm), width=3),
            
        ],
        style={"height": "300px"}
        )
    

    @dash_app.callback(
        Output('graphs', 'children'),
        [
            Input('symbol', 'value'),
            Input('my-date-picker-range', 'start_date'),
            Input('my-date-picker-range', 'end_date')
        ],
        prevent_initial_call=True
    )

    def create_technical_graphs(symbol,start_date, end_date):
        data, stock = data_ingestor.get_stock_data(symbol = symbol, start = start_date, end = end_date)

        data_math = data_prepare.calculate_mathematical_term(data)

        return (
            [
                dbc.Row(du.make_px_graph(f"{symbol} Close Price Over the Years",data.index,['Close'],data)),
                html.Br(),
                dbc.Row(du.make_px_graph(f"{symbol} Volume traded",data.index,['Volume'], df = data)),
                html.Br(),
                dbc.Row(du.make_px_graph(f"{symbol} Simple Moving Averages (50 Days and 200 Days)",data_math.index,["Close", "sma_50", "sma_200"], data_math)),
                html.Br(),
                dbc.Row(du.plot_MACD(data_math,f"{symbol} MACD(Moving Average Convergence Divergence) Graph")),
                html.Br(),
                dbc.Row(du.plot_rsi(data_math,f"{symbol} RSI(Relative Strength Index) Graph")),
                html.Br(),
            ]
        )


    return dash_app


   