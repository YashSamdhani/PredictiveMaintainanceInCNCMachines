import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
import pickle
import numpy as np
from dash.dependencies import Input, Output
import dash_daq as daq

# Load datasets
test_df = pd.read_csv("E:/test.csv")  # Update path
actual_df = pd.read_csv("E:/Data.csv")  # Load actual values

# Load the model
with open("E:/model.pkl", "rb") as f:
    model = pickle.load(f)

def make_prediction(data):
    data = np.array(data).reshape(1, -1)  # Reshape for model
    prediction = float(model.predict(data).item())  # Ensure output is a scalar
    print(f"Model Prediction: {prediction}")  # Print prediction to terminal
    return prediction

# Get max Tool Wear value from actual dataset
max_tool_wear = actual_df["tool_wear"].max()

# Initialize Dash app
app = dash.Dash(__name__)
index = 0  # Keep track of row index

app.layout = html.Div([
    html.H1("Tool Wear Prediction Dashboard", style={"textAlign": "center", "font-family": "Poppins", "color": "#ffffff", "backgroundColor": "#121212", "padding": "20px", "borderRadius": "10px"}),
    
    html.Div([
        daq.Gauge(
            id="gauge-chart",
            min=0,
            max=100,
            value=0,
            label="Tool Wear (%)",
            showCurrentValue=True,
            units="%",
            size=350,
            color={"gradient": True, "ranges": {"green": [0, 30], "yellow": [30, 70], "red": [70, 100]}},
            scale={"start": 0, "interval": 10, "labelInterval": 1},
            style={"gridColumn": "span 3", "backgroundColor": "#1e1e1e", "borderRadius": "10px", "padding": "15px", "textAlign": "center"}
        )
    ], style={"display": "grid", "grid-template-columns": "repeat(3, 1fr)", "gap": "20px", "padding": "20px"}),
    
    html.Div([
        dcc.Graph(id="live-chart", animate=True)
    ], style={"backgroundColor": "#1e1e1e", "borderRadius": "10px", "padding": "15px", "boxShadow": "0px 4px 10px rgba(255, 255, 255, 0.1)"}),
    
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
], style={"backgroundColor": "#121212", "color": "#ffffff", "minHeight": "100vh", "padding": "20px", "font-family": "Poppins"})

@app.callback(
    [Output("gauge-chart", "value"), Output("live-chart", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_charts(n):
    global index
    if index >= len(test_df):
        index = 0  # Restart when reaching the end of data
    
    latest_data = test_df.iloc[index].values  # Get row by row for real-time effect
    prediction = make_prediction(latest_data)
    scaled_prediction = (prediction / max_tool_wear) * 100 if max_tool_wear else 0  # Scale to 0-100
    
    fig = px.line(test_df.iloc[:index+1], x=test_df.index[:index+1], y=test_df.columns, title="Live Data Over Time", template="plotly_dark")
    index += 1
    return scaled_prediction, fig

if __name__ == "__main__":
    app.run_server(debug=True)
