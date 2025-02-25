from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.payload import BinaryPayloadBuilder, BinaryPayloadDecoder
from pymodbus.constants import Endian
from pymodbus.client.sync import ModbusTcpClient

import pandas as pd
import time
import joblib
import threading
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np

# File Paths
test_csv_path = "E:/test.csv"
data_csv_path = "E:/data.csv"
model_path = "E:/model.pkl"

# Load max tool_wear for scoring (assumes 'tool_wear' column exists)
data_df = pd.read_csv(data_csv_path)
max_tool_wear = data_df['tool_wear'].max()

class ModbusServerThread:
    def __init__(self, csv_path, port=5020):
        self.data = pd.read_csv(csv_path)
        self.port = port
        self.start_time = time.time()
        self.total_data_sent = 0
        self.columns = self.data.columns.tolist()
        self.n = len(self.columns)
        # Each float value will be stored in 2 registers
        self.total_registers = 2 * self.n
        # Create datastore with a sequential block for holding registers
        store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0]*10),  # dummy
            co=ModbusSequentialDataBlock(0, [0]*10),  # dummy
            hr=ModbusSequentialDataBlock(0, [0]*self.total_registers),
            ir=ModbusSequentialDataBlock(0, [0]*10)
        )
        self.context = ModbusServerContext(slaves=store, single=True)
        # Map each CSV column to a starting register address (0, 2, 4, …)
        self.column_address = {col: i*2 for i, col in enumerate(self.columns)}

    def start_server(self):
        """
        Starts the Modbus TCP server on the specified port.
        This call is blocking so run it in a separate thread.
        """
        StartTcpServer(self.context, address=("localhost", self.port))

    def update_data(self):
        """
        Loops through the CSV data and, for each row, encodes each column's
        float value into two registers and writes them to the holding registers.
        """
        for _, row in self.data.iterrows():
            for col in self.columns:
                value = float(row[col])
                # Encode the float into two registers
                builder = BinaryPayloadBuilder(byteorder=Endian.Big, wordorder=Endian.Big)
                builder.add_32bit_float(value)
                registers = builder.to_registers()  # returns a list of two registers
                address = self.column_address[col]
                # Write the two registers for this column (function code 3: holding registers)
                self.context[0].setValues(3, address, registers)
            self.total_data_sent += 1
            time.sleep(1)  # simulate real-time data streaming

    def get_statistics(self):
        uptime = time.time() - self.start_time
        data_rate = self.total_data_sent / uptime if uptime > 0 else 0
        stats = {
            "Uptime (seconds)": round(uptime, 2),
            "Total Data Sent": self.total_data_sent,
            "Data Transfer Rate (rows/sec)": round(data_rate, 2)
        }
        print("DEBUG: Statistics dictionary:", stats)  # Debug line to verify keys
        return stats


class ModbusClient:
    def __init__(self, port=5020):
        self.port = port
        self.client = ModbusTcpClient('localhost', port=self.port)
        self.model = joblib.load(model_path)
        self.data = pd.read_csv(test_csv_path)
        self.columns = self.data.columns.tolist()
        self.n = len(self.columns)
        self.total_registers = 2 * self.n
        # Map each column to a starting register address
        self.column_address = {col: i*2 for i, col in enumerate(self.columns)}
        self.received_data = {col: [] for col in self.columns}
        self.model_outputs = []
        self.latency_measurements = []

    def connect(self):
        self.client.connect()

    def fetch_data(self):
        """
        In a loop, read all holding registers from the Modbus server,
        decode the float values for each column, and then perform a model
        prediction on the row.
        """
        while True:
            start_time = time.time()
            # Read all holding registers that hold our data
            result = self.client.read_holding_registers(0, self.total_registers)
            if result.isError():
                print("Error reading registers")
                time.sleep(1)
                continue
            registers = result.registers
            row = {}
            # Decode each float value from its two registers
            for i, col in enumerate(self.columns):
                reg_pair = registers[2*i:2*i+2]
                decoder = BinaryPayloadDecoder.fromRegisters(reg_pair, byteorder=Endian.Big, wordorder=Endian.Big)
                value = decoder.decode_32bit_float()
                row[col] = value
                self.received_data[col].append(value)
            latency = time.time() - start_time
            self.latency_measurements.append(latency)
            # Perform model prediction on the current row
            df_row = pd.DataFrame([row])
            prediction = self.model.predict(df_row)[0]
            self.model_outputs.append(prediction)
            time.sleep(1)

    def get_average_latency(self):
        return round(np.mean(self.latency_measurements), 4) if self.latency_measurements else 0

    def start_dashboard(self):
        """
        Creates a Dash dashboard that displays live graphs for each CSV column
        and a gauge showing the model’s latest prediction (scaled by max_tool_wear).
        """
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
            for col in self.columns:
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

def compare_servers(modbus_stats, latency):
    data_rate = modbus_stats.get("Data Transfer Rate (rows/sec)", "N/A")
    print("\nServer Comparison Analysis:")
    print(f"Modbus: Latency={latency}s, Data Rate={data_rate} rows/sec, Security=Basic")

if __name__ == "__main__":
    # Create Modbus server and client objects
    modbus_server = ModbusServerThread(test_csv_path, port=5020)
    modbus_client = ModbusClient(port=5020)

    # Start the Modbus TCP server in a separate thread
    server_thread = threading.Thread(target=modbus_server.start_server)
    # Start the data update loop (simulates sending CSV rows) in another thread
    updater_thread = threading.Thread(target=modbus_server.update_data)
    # Start the Modbus client data fetching loop in its own thread
    client_thread = threading.Thread(target=modbus_client.fetch_data)
    # Start the dashboard in a separate thread
    dashboard_thread = threading.Thread(target=modbus_client.start_dashboard)

    server_thread.start()
    time.sleep(2)  # Give the server time to start
    modbus_client.connect()
    updater_thread.start()
    client_thread.start()
    dashboard_thread.start()

    time.sleep(5)  # Allow some data transfer before analysis
    stats = modbus_server.get_statistics()
    avg_latency = modbus_client.get_average_latency()
    compare_servers(stats, avg_latency)
