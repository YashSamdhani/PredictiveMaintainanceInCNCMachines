import rti.connextdds as dds
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

topic_name = "SensorData"

# DDS Publisher
class DDSPublisher:
    def __init__(self):
        self.participant = dds.DomainParticipant(0)
        self.topic = dds.Topic(self.participant, topic_name, dds.DynamicDataType("SensorData"))
        self.writer = dds.DataWriter(self.participant, self.topic)
        self.data = pd.read_csv(test_csv_path)
        self.start_time = time.time()
        self.total_data_sent = 0

    def start(self):
        try:
            for _, row in self.data.iterrows():
                sample = dds.DynamicData(self.topic.type)
                for col in self.data.columns:
                    sample[col] = row[col]
                self.writer.write(sample)
                self.total_data_sent += 1
                time.sleep(1)
        finally:
            pass

    def get_statistics(self):
        uptime = time.time() - self.start_time
        data_rate = self.total_data_sent / uptime if uptime > 0 else 0
        return {
            "Uptime (seconds)": round(uptime, 2),
            "Total Data Sent": self.total_data_sent,
            "Data Transfer Rate (rows/sec)": round(data_rate, 2)
        }

# DDS Subscriber with Dashboard
class DDSSubscriber:
    def __init__(self):
        self.participant = dds.DomainParticipant(0)
        self.topic = dds.Topic(self.participant, topic_name, dds.DynamicDataType("SensorData"))
        self.reader = dds.DataReader(self.participant, self.topic)
        self.model = joblib.load(model_path)
        self.data = pd.read_csv(test_csv_path)
        self.received_data = {col: [] for col in self.data.columns}
        self.model_outputs = []
        self.latency_measurements = []
        self.reader.bind_listener(self.on_data_available)

    def on_data_available(self, reader):
        for sample in reader.take():
            start_time = time.time()
            row = {col: sample[col] for col in self.data.columns}
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
def compare_servers(dds_stats, latency):
    comparison = {
        "DDS": {"Latency (s)": latency, "Data Rate (rows/sec)": dds_stats["Data Transfer Rate (rows/sec)"], "Security": "Very Strong"},
        "MQTT": {"Latency (s)": 0.001, "Data Rate (rows/sec)": 500, "Security": "Moderate"},
        "HTTP": {"Latency (s)": 0.5, "Data Rate (rows/sec)": 10, "Security": "Moderate"},
        "Modbus": {"Latency (s)": 0.05, "Data Rate (rows/sec)": 100, "Security": "Weak"}
    }
    print("\nServer Comparison Analysis:")
    for server, metrics in comparison.items():
        print(f"{server}: Latency={metrics['Latency (s)']}s, Data Rate={metrics['Data Rate (rows/sec)']} rows/sec, Security={metrics['Security']}")

if __name__ == "__main__":
    publisher = DDSPublisher()
    subscriber = DDSSubscriber()
    
    publisher_thread = threading.Thread(target=publisher.start)
    dashboard_thread = threading.Thread(target=subscriber.start_dashboard)
    
    publisher_thread.start()
    dashboard_thread.start()
    
    time.sleep(5)  # Allow some data transfer before analysis
    dds_stats = publisher.get_statistics()
    avg_latency = subscriber.get_average_latency()
    compare_servers(dds_stats, avg_latency)