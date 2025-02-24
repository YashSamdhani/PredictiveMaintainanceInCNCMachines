import paho.mqtt.client as mqtt
import pandas as pd
import time
import joblib
import threading
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
import json

# File Paths
test_csv_path = "E:/test.csv"
data_csv_path = "E:/data.csv"
model_path = "E:/model.pkl"

# Load max tool_wear for scoring
data_df = pd.read_csv(data_csv_path)
max_tool_wear = data_df['tool_wear'].max()

topic = "sensor/data"

# MQTT Server (Broker)
class MQTTServer:
    def __init__(self):
        self.client = mqtt.Client()
        self.data = pd.read_csv(test_csv_path)
        self.start_time = time.time()
        self.total_data_sent = 0
        self.client.on_connect = self.on_connect

    def on_connect(self, client, userdata, flags, rc):
        print("MQTT Broker Connected")

    def start(self):
        self.client.connect("localhost", 1883, 60)
        self.client.loop_start()
        try:
            for _, row in self.data.iterrows():
                self.client.publish(topic, json.dumps(row.to_dict()))
                self.total_data_sent += 1
                time.sleep(1)
        finally:
            self.client.loop_stop()

    def get_statistics(self):
        uptime = time.time() - self.start_time
        data_rate = self.total_data_sent / uptime if uptime > 0 else 0
        return {
            "Uptime (seconds)": round(uptime, 2),
            "Total Data Sent": self.total_data_sent,
            "Data Transfer Rate (rows/sec)": round(data_rate, 2)
        }

# MQTT Client with Dashboard
class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.model = joblib.load(model_path)
        self.data = pd.read_csv(test_csv_path)
        self.received_data = {col: [] for col in self.data.columns}
        self.model_outputs = []
        self.latency_measurements = []
        self.client.on_message = self.on_message
        self.client.on_connect = self.on_connect

    def on_connect(self, client, userdata, flags, rc):
        print("MQTT Client Connected")
        self.client.subscribe(topic)

    def on_message(self, client, userdata, msg):
        start_time = time.time()
        row = json.loads(msg.payload.decode())
        latency = time.time() - start_time
        self.latency_measurements.append(latency)
        for col in self.data.columns:
            self.received_data[col].append(row[col])
        
        # Model prediction
        df_row = pd.DataFrame([row])
        prediction = self.model.predict(df_row)[0]
        self.model_outputs.append(prediction)
    
    def get_average_latency(self):
        return round(np.mean(self.latency_measurements), 4) if self.latency_measurements else 0
    
    def start_dashboard(self):
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("Real-Time Data Dashboard", style={'textAlign': 'center', 'color': 'white'}),
            html.Div(id='graphs-container'),
            dcc.Graph(id='gauge-chart')
        ], style={'backgroundColor': '#222', 'color': 'white', 'padding': '20px'})
        
        @app.callback(
            Output('graphs-container', 'children'),
            Input('gauge-chart', 'id')
        )
        def update_graphs(_):
            graphs = []
            for col in self.data.columns:
                graphs.append(dcc.Graph(
                    figure={
                        'data': [go.Scatter(y=self.received_data[col], mode='lines', name=col)],
                        'layout': go.Layout(title=col, paper_bgcolor='#222', font={'color': 'white'})
                    }
                ))
            return graphs
        
        @app.callback(
            Output('gauge-chart', 'figure'),
            Input('gauge-chart', 'id')
        )
        def update_gauge(_):
            if self.model_outputs:
                score = (self.model_outputs[-1] / max_tool_wear) * 100
                color = 'green' if score < 33 else 'yellow' if score < 66 else 'red'
                return go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color}},
                    title={'text': "Tool Wear Severity"}
                ))
            return go.Figure()
        
        app.run_server(debug=True)

# Server Comparison Analysis
def compare_servers(mqtt_stats, latency):
    comparison = {
        "MQTT": {"Latency (s)": latency, "Data Rate (rows/sec)": mqtt_stats["Data Transfer Rate (rows/sec)"], "Security": "Moderate"}
    }
    print("\nServer Comparison Analysis:")
    for server, metrics in comparison.items():
        print(f"{server}: Latency={metrics['Latency (s)']}s, Data Rate={metrics['Data Rate (rows/sec)']} rows/sec, Security={metrics['Security']}")

if __name__ == "__main__":
    server = MQTTServer()
    client = MQTTClient()
    
    server_thread = threading.Thread(target=server.start)
    client_thread = threading.Thread(target=client.client.loop_forever)
    dashboard_thread = threading.Thread(target=client.start_dashboard)
    
    server_thread.start()
    time.sleep(2)  # Give time for server to start
    client.client.connect("localhost", 1883, 60)
    client_thread.start()
    dashboard_thread.start()
    
    time.sleep(5)  # Allow some data transfer before analysis
    mqtt_stats = server.get_statistics()
    avg_latency = client.get_average_latency()
    compare_servers(mqtt_stats, avg_latency)
