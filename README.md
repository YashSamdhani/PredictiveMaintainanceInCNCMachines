# Predictive Maintenance in CNC Machines: A Comprehensive Framework

**Project Thesis Report**  
Submitted by: **Yashowardhan Samdhani (RA2111026010151)**  
Institution: SRM Institute of Science and Technology, Kattankulathur  
Submission Date: March 2025  
Under the Guidance of: **Prof. Dr. Christoph Reich**

---

## Table of Contents

- [Abstract](#abstract)
- [Objective](#objective)
- [Literature Review](#literature-review)
- [Proposed Methodology](#proposed-methodology)
- [Machine Learning Model](#machine-learning-model)
- [Dashboard Development](#dashboard-development)
- [Networking and Data Transfer](#networking-and-data-transfer)
- [Future Work](#future-work)

---

## Abstract

Predictive Maintenance in CNC machines is a crucial advancement for reducing downtime and increasing operational efficiency in industrial automation. This project:
- **Develops a robust predictive maintenance framework** by integrating machine learning models with real-time monitoring.
- **Evaluates and compares** traditional (Linear Regression) and advanced (LSTM-based Neural Network) models for tool wear prediction.
- **Implements an interactive dashboard** using Python Dash for live visualization of sensor data.
- **Assesses industrial communication protocols** (OPC UA, MQTT, and Modbus) to ensure secure, low-latency data transfer.

The comprehensive approach provides a practical roadmap for implementing data-driven maintenance strategies in CNC environments.

---

## Objective

### General Objective
Develop a scalable, adaptable, and robust predictive maintenance system for CNC machines that integrates:
- Advanced machine learning techniques
- Real-time data visualization
- Reliable network communication protocols

### Specific Objectives
- **Machine Learning Analysis:** Compare Linear Regression with Neural Network (LSTM) architectures for accurate tool wear prediction.
- **Dashboard Development:** Create a user-friendly, real-time monitoring interface that visualizes sensor readings, predictions, and alerts.
- **Networking Evaluation:** Test and compare OPC UA, MQTT, and Modbus protocols to determine the most efficient method for real-time data transmission.
- **System Integration:** Seamlessly integrate data preprocessing, model training, dashboard visualization, and network communication into a unified system.

### Scope and Constraints
- **Machine Learning:** Focus is limited to Linear Regression and LSTM-based Neural Networks.
- **Dashboard:** Initial trials with Node-RED were replaced by Python Dash due to connectivity and stability issues.
- **Networking:** Only OPC UA, MQTT, and Modbus are evaluated, emphasizing performance metrics such as latency and data throughput.
- **Deployment:** The system is simulated using industrial datasets; full-scale industrial deployment is a future extension.

---

## Literature Review

A detailed review was conducted covering:
- **Deep Learning for Tool Wear:** Utilization of repositories like Keras Detect Tool Wear to understand CNN-based and LSTM models.
- **Benchmark Datasets:** Analysis of standard datasets such as PHM Data Challenge 2010 and Uniwear, highlighting their advantages in replicating real-world machining conditions.
- **UI Frameworks:** Comparison of Node-RED and Python Dash, examining aspects like real-time data integration, performance stability, and ease of customization.
- **Industrial Communication Protocols:** Theoretical and empirical comparisons among OPC UA, MQTT, and Modbus, assessing features like security, scalability, and latency.

These studies provide the foundation for selecting appropriate tools and techniques for the project.

---

## Proposed Methodology

### 1. Dataset Preparation
- **Data Collection:** Sensor data is gathered from CNC machines using datasets like Uniwear.
- **Data Cleaning and Normalization:** Removal of unnecessary columns (e.g., 'Unnamed: 0', 'experiment_tag') and application of StandardScaler.
- **Data Splitting:** Partitioning data into training and testing sets based on unique experiment tags.

### 2. Machine Learning Model Comparative Analysis
- **Model Selection:** Evaluation of Linear Regression vs. Neural Network models.
- **Training:** Implementation using TensorFlow/Keras with an LSTM layer to capture temporal dependencies.
- **Evaluation Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score are used for performance assessment.
- **Visualization:** Loss curves and scatter plots comparing predicted versus actual tool wear values.

### 3. Real-Time Dashboard Development
- **Final Implementation:** A robust Python Dash-based dashboard was developed with:
  - Dynamic time-series graphs
  - An interactive gauge indicator for real-time tool wear severity
  - A dark-themed, high-contrast UI for enhanced readability.

### 4. Network Communication Protocol Evaluation
- **Protocols Tested:** OPC UA, MQTT, and Modbus.
- **Performance Metrics:** Latency, data transfer rate, and reliability were measured.
- **Results:** OPC UA emerged as the most secure and scalable option, despite MQTT’s lightweight nature and Modbus’s legacy system compatibility.

---

## Machine Learning Model

### Model Architecture
- **LSTM Layer:** Captures sequential dependencies in sensor data.
- **Dropout Regularization:** Prevents overfitting.
- **Dense Layers:** Additional dense layers for non-linear feature extraction.
- **Output:** Single neuron regression output for tool wear prediction.

### Training and Evaluation
- **Compilation:** Using Adam optimizer, MSE loss function, and MAE as an auxiliary metric.
- **Training:** Model is trained for 100 epochs with a batch size of 32.
- **Performance Metrics:** Training and testing metrics, including MSE, MAE, and R² score, validate the model’s predictive capability.

---

## Dashboard Development

### Initial Trials with Node-RED
- **Challenges:** Persistent database connectivity issues (e.g., MySQL error “getaddrinfo ENOTFOUND”) and performance instability led to frequent crashes and memory leaks.

### Transition to Python Dash
- **Design:**
  - A dark-themed UI with clear headers and dynamic graphs.
  - Real-time updates using automated callbacks.
- **Key Features:**
  - Dynamic Graphs: Time-series plots for various sensor measurements.
  - Gauge Indicator: Visualizes the severity of tool wear with color-coded alerts (green, yellow, red).
  - Interactivity: Continuous updating ensures operators have immediate, actionable insights.
- **Outcome:** A stable, flexible, and customizable dashboard that integrates seamlessly with the predictive model.

---

## Networking and Data Transfer

### Protocols Evaluated
- **OPC UA:** Service-oriented architecture with strong security features.
- **MQTT:** Lightweight, broker-based publish/subscribe model ideal for IoT.
- **Modbus:** Simple client/server mechanism best suited for legacy systems.

---

## Future Work

- **IoT-Based Data Acquisition:** Integrate edge computing and IoT sensors for real-time monitoring.
- **Advanced Machine Learning:** Implement Explainable AI (XAI) and AutoML for dynamic model tuning.
- **Dashboard Enhancements:** Add augmented reality (AR) interfaces and mobile-friendly designs.
- **Optimizing Communication:** Investigate hybrid protocols and deploy distributed data processing.

