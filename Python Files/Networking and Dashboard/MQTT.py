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

# MQTT Configuration
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "sensor/data"

# File Paths
test_csv_path = "E:/test.csv"
data_csv_path = "E:/data.csv"
model_path = "E:/model.pkl"

# Load max tool_wear for scoring
data_df = pd.read_csv(data_csv_path)
max_tool_wear = data_df['tool_wear'].max()

def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker!" if rc == 0 else f"Failed to connect, return code {rc}")

# MQTT Server
class MQTTServer:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.data = pd.read_csv(test_csv_path)
        self.start_time = time.time()
        self.total_data_sent = 0

    def start(self):
        self.client.loop_start()
        try:
            while True:
                for _, row in self.data.iterrows():
                    payload = row.to_json()
                    self.client.publish(MQTT_TOPIC, payload)
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

# MQTT Client
class MQTTClient:
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = on_connect
        self.client.on_message = self.on_message
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.model = joblib.load(model_path)
        self.received_data = {col: [] for col in pd.read_csv(test_csv_path).columns}
        self.model_outputs = []
        self.latency_measurements = []
        self.client.subscribe(MQTT_TOPIC)
        self.client.loop_start()
    
    def on_message(self, client, userdata, msg):
        start_time = time.time()
        row = json.loads(msg.payload.decode())
        latency = time.time() - start_time
        self.latency_measurements.append(latency)
        for col, value in row.items():
            self.received_data[col].append(value)
        
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
            for col in self.received_data.keys():
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

if __name__ == "__main__":
    server = MQTTServer()
    client = MQTTClient()
    
    server_thread = threading.Thread(target=server.start)
    dashboard_thread = threading.Thread(target=client.start_dashboard)
    
    server_thread.start()
    time.sleep(2)  # Allow time for server to start
    dashboard_thread.start()
    
    time.sleep(5)  # Allow some data transfer before analysis
    mqtt_stats = server.get_statistics()
    avg_latency = client.get_average_latency()
    
    print("\nServer Comparison Analysis:")
    print(f"MQTT: Latency={avg_latency}s, Data Rate={mqtt_stats['Data Transfer Rate (rows/sec)']} rows/sec, Security=Moderate")
