# 🏙️ Air Pollution Dispersion Prediction using ANN, CNN & CART

This project investigates the dispersion of **SOx, NOx, and PM10** pollutants in Ranchi, India, due to **vehicular emissions**, using **machine learning models**:  
- 🧠 Artificial Neural Networks (ANN)  
- 📦 Convolutional Neural Networks (CNN)  
- 🌳 Classification and Regression Trees (CART)

📍 **Location**: Ranchi, Jharkhand, India  
📅 **Duration**: 2019–2023  
👨‍💻 **Technologies**: MATLAB, Environmental Data Analysis, Machine Learning

---

## 🔍 Problem Statement

Vehicular emissions are a significant contributor to urban air pollution in growing Indian cities. This project models how **meteorological parameters** influence the dispersion of key pollutants. It aims to:

✅ Forecast pollutant levels using supervised ML models  
✅ Compare model performance using metrics like **RMSE**, **R²**, **MAE**  
✅ Identify pollution hotspots for urban planning interventions  

---

## 📊 Dataset Overview

**📦 Dataset Size**: 1,310 samples  
**⏳ Time Span**: Daily measurements, 104 days/year over 5 years  
**📌 Features** (Inputs):  
- RH (%) – Relative Humidity  
- WS (m/s) – Wind Speed  
- WD (°) – Wind Direction  
- SR (W/m²) – Solar Radiation  
- BP (mmHg) – Barometric Pressure  
- VWS (m/s) – Vertical Wind Speed  
- AT (°C) – Air Temperature  
- RF (mm) – Rainfall  

**🎯 Target Variables**:  
- SOx (μg/m³)  
- NOx (μg/m³)  
- PM10 (μg/m³)  

---

## 🛠️ Methodology

### ✅ ANN (Artificial Neural Network)
- Feedforward MLP with 2 hidden layers (20 & 10 neurons)
- Bayesian Regularization Training (`trainbr`)
- Performance Function: Mean Squared Error (MSE)

### ✅ CNN (Convolutional Neural Network)
- Data reshaped as feature maps (8×1×1×1310)
- Convolutional Layer + Fully Connected Layers
- Optimizer: Adam | Epochs: 200 | Batch Size: 32

### ✅ CART (Classification and Regression Tree)
- Regression tree using MSE splitting
- MinLeafSize = 5
- Surrogate splits enabled

---

## 📈 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Squared Error – measures prediction scatter |
| **R²**   | Coefficient of Determination – indicates fit quality |
| **MAE**  | Mean Absolute Error – average magnitude of errors |

---

## ✅ Model Results Summary

| Pollutant | Best Model | RMSE  | R²     | MAE   |
|-----------|------------|-------|--------|-------|
| **SOx**   | CART       | 8.130 | 0.7377 | 5.832 |
| **NOx**   | CNN        | 7.425 | 0.8448 | 4.912 |
| **PM10**  | CNN        | 10.01 | 0.8923 | 6.315 |

📌 **Note**: CNN achieved the best overall performance, especially in NOx and PM10 forecasting.

---

## 📊 Visual Results

### 🔹 ANN – Predicted vs. Observed
![ANN SOx](./results/plots/ann_sox.png)

### 🔹 CNN – Predicted vs. Observed
![CNN NOx](./results/plots/cnn_nox.png)

### 🔹 CART – Predicted vs. Observed
![CART PM10](./results/plots/cart_pm10.png)

---

## 🧠 Key Insights

- CNN consistently outperformed ANN and CART for **NOx and PM10**
- CART delivered best results for **SOx**, highlighting its interpretability and effectiveness with limited data
- ANN was close to CNN for PM10 but lagged in other areas

---

## 📂 Project Structure

```bash
air-pollution-dispersion-model/
│
├── data/                  # Raw data (CSV)
│   ├── Data.csv
│   └── WB014.csv
│
├── code/                  # MATLAB model scripts
│   ├── ann_model.m
│   ├── cnn_model.m
│   ├── cart_model.m
│
├── results/
│   ├── plots/             # Visual outputs
│   └── metrics.txt        # Summary of metrics
│
├── docs/
│   ├── Report.pdf         # Final report
│   └── content.docx       # Editable version
│
└── README.md              # This file
