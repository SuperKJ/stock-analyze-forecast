import dash_bootstrap_components as dbc
from dash import html, dash_table, dcc, callback, Input, Output, State
import plotly.express as px
from dash import dcc
import numpy as np
import plotly.graph_objects as go

def create_nav():
    return dbc.Nav([
                html.Div([
                    html.A("Stock Dashboard", className="navbar-brand", href="/"),
                    html.Ul([
                        html.Li(html.A("Analyze", href="/analyze/", className="nav-link"), className="nav-item"),
                        html.Li(html.A("Train", href="/train/", className="nav-link"), className="nav-item"),
                        html.Li(html.A("Forecast", href="/forecast/", className="nav-link"), className="nav-item"),
                    ], className="navbar-nav me-auto")
                ], className="container-fluid")
            ], className="navbar navbar-expand-lg navbar-dark bg-dark"),

def make_info_card(title, value):
    return dbc.Card(
        dbc.CardBody([
            dbc.CardHeader(title, style={"font-weight": "bolder"}),
            dbc.CardBody(value)
        ]),
        className="h-50",
    )
    
def make_px_graph(title,x,y, df = None):
    fig = px.line(df, x=x, y=y)
    fig.update_layout(
        template='plotly_dark', 
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_rangeslider_thickness=0.1,
        xaxis_rangeslider_bgcolor='rgba(0, 0, 0, 0.1)',
        xaxis_rangeslider_bordercolor='rgba(0, 0, 0, 0.1)',  
         
    )

    return dbc.Card(
        dbc.CardBody([
            dbc.CardHeader(title, style={"font-weight": "bolder"}),
            dbc.CardBody([dcc.Graph(figure=fig)]),
        ]),
        className="h-50",
    )


def plot_MACD(df,title):
    """
    Plot MACD using Plotly.
    Parameters:
    df (DataFrame): DataFrame containing stock data.
    """
    colors = np.where(df['MACD_Diff'] > 0, 'green', 'red')
    fig = go.Figure()

    fig.add_bar(
        x=df.index,
        y=df.MACD_Diff,
        name='MACD Difference',
        marker=dict(color=colors),
    )

    fig.add_trace(go.Scatter(
                x=df.index,
                y=df['MACD'],
                name='MACD',
                opacity=0.8,
                line=dict(color='blue') 
                ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Signal_Line'],
        name='Signal Line',
        opacity=0.8,
        line=dict(color='grey')  # <-- Use this for line color
    ))

    fig.update_layout(
        template='plotly_dark', 
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="MACD Difference",
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.05,
        xaxis_rangeslider_bgcolor='rgba(0, 0, 0, 0.1)',
        xaxis_rangeslider_bordercolor='rgba(0, 0, 0, 0.1)',   
    )

    return dbc.Card(
        dbc.CardBody([
            dbc.CardHeader(title, style={"font-weight": "bolder"}),
            dbc.CardBody([dcc.Graph(figure=fig)]),
            dbc.CardFooter("MACD helps show if a stock’s trend is gaining strength or slowing down. When the lines cross, it can signal a good time to buy or sell.")
        ]),
        className="h-50",
    )



def plot_rsi(df,title):
    """
    Plot RSI using Plotly.
    Parameters:
    df (DataFrame): DataFrame containing stock data.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['RSI'],
        name='RSI',
        line=dict(color='blue')
    ))

    fig.add_hline(y=70, line_color='red', line_dash='dash', annotation_text="Overbought", annotation_position="bottom right")
    fig.add_hline(y=30, line_color='green', line_dash='dash', annotation_text="Oversold", annotation_position="top right")

    fig.update_layout(
        template='plotly_dark', 
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Date",
        yaxis_title="RSI",
        xaxis_rangeslider_visible=True,
        xaxis_rangeslider_thickness=0.05,
        xaxis_rangeslider_bgcolor='rgba(0, 0, 0, 0.1)',
        xaxis_rangeslider_bordercolor='rgba(0, 0, 0, 0.1)',   
    )

    return dbc.Card(
        dbc.CardBody([
            dbc.CardHeader(title, style={"font-weight": "bolder"}),
            dbc.CardBody([dcc.Graph(figure=fig)]),
            dbc.CardFooter("RSI tells if a stock has been going up or down too quickly.If it’s too high, the stock might be overvalued (time to sell); if it’s too low, it might be undervalued (time to buy).")
        ]),
        className="h-50",
    )

def plot_train_pred_forecast(rmse, mae,r2,fig, forecast_data, symbol):
    return html.Div([
                dbc.Row(
                    [
                        dbc.Col(make_info_card("Root Mean Squared Error", rmse)),
                        dbc.Col(make_info_card("Mean Absolute Error",mae)),
                        dbc.Col(make_info_card("R2 Score", r2)),
                        
                    ],
                    style={"height": "300px"}
                    ),

                dbc.Row([
                    dbc.Card(
                            dbc.CardBody([
                                            dbc.CardHeader("Actual Vs Predicted Prices", style={"font-weight": "bolder"}),
                                            dbc.CardBody([dcc.Graph(figure=fig)]),
                                        ]),
                                        
                            )
                ]),
                html.Br(),
                html.Hr(),
                html.Br(),

                html.H4("Forecast"),

                dbc.Row([ 
                    dbc.Col(
                        dbc.Card(
                            # dbc.CardBody([
                                # dbc.CardHeader("Forecasted Close Price Graph", style={"font-weight": "bolder"}),
                                # dbc.CardBody([
                                    make_px_graph("Forecasted Close Price", forecast_data['Date'], "Close", forecast_data),
                                # ])
                            # ]),
                            style={"height": "100%"}
                        ),
                        width=6  # adjust width as needed
                    ),
                    dbc.Col( 
                        dbc.Card(
                            dbc.CardBody([
                                dbc.CardHeader("Forecasted Close Price Table", style={"font-weight": "bolder"}),
                                dbc.CardBody([
                                    dash_table.DataTable(
                                        data=forecast_data.reset_index().to_dict('records'),
                                        columns=[{"name": i, "id": i} for i in forecast_data.reset_index().columns],
                                        style_table={'overflowX': 'auto'},
                                        style_header={
                                            'backgroundColor': 'rgb(30, 30, 30)',
                                            'color': 'white',
                                            'fontWeight': 'bold'
                                        },
                                        style_cell={
                                            'backgroundColor': 'rgb(50, 50, 50)',
                                            'color': 'white',
                                            'textAlign': 'left'
                                        },
                                        page_size=15
                                    )
                                ])
                            ])
                        ),
                        width=6
                    )
                ])
            ])

