# 🌦️ Weather Prediction and Analysis with Machine Learning

This project presents a comprehensive pipeline for analyzing weather data using machine learning and deep learning models. It includes data preprocessing, exploratory analysis, classification using traditional ML algorithms, clustering techniques, and a neural network for rainfall prediction.

---

## 🔍 Project Overview

This repository contains the following main components:

### ✅ Web Scraping  
Scrapes weather data from [timeanddate.com](https://www.timeanddate.com) for use in various ML tasks.

### ✅ Classification  
Implements supervised learning models such as:
- Logistic Regression  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  
to classify weather conditions based on meteorological data.

### ✅ Clustering  
Uses unsupervised learning techniques:
- KMeans  
- DBSCAN  
to discover hidden patterns in weather observations.

### ✅ Neural Network  
A **PyTorch-based binary classifier** trained to predict rainfall using weighted loss to handle class imbalance.

### ✅ Time Series Analysis and Forecasting  
Applies **ARIMA** and **SARIMA** models for temperature forecasting using 3 years of hourly data from 188 cities.

### ✅ Regression  
Predicts continuous weather metrics (e.g., temperature, humidity) using:
- Linear Regression  
- Random Forest Regressor  

---

## 📁 Files

| File                          | Description                                                       |
|-------------------------------|-------------------------------------------------------------------|
| `Final_project_Classification.ipynb` | Weather classification using multiple ML models                |
| `Final_pr_Clustering.ipynb`           | Exploratory clustering analysis                                |
| `Final_prj_neural_network.py`        | Rain prediction using a PyTorch neural network                 |
| `Final_project.ipynb`                | Time Series analysis and regression using various ML models    |

---

## 🔗 Access to Key Notebooks

You can explore the **Web Scraping**, **Time Series**, and **Regression** parts of the project directly on Google Colab:

👉 [Open in Google Colab](https://colab.research.google.com/drive/1Hn1qtp6aV7zcgJ8jOhdn8kSpdWgKsXVb?usp=sharing)

> **Note:** The other notebooks (Classification, Clustering, Neural Network) must be downloaded from this repository.

---

## 🛠️ Tools & Libraries

- Python, Pandas, NumPy, Seaborn, Matplotlib  
- Scikit-learn  
- PyTorch  
- Google Colab & Google Drive (for dataset access)

---

## 📊 Dataset

The dataset (`weather_2.csv`) is stored in Google Drive and includes the following features:

- Temperature  
- Humidity  
- Wind Speed  
- Visibility  
- UV Index  
- Precipitation  

---

## 📈 Model Performance

- The PyTorch neural network achieves strong binary classification performance for rainfall prediction.
- **Weighted loss** and **feature normalization** were used to address class imbalance and improve learning.
- Traditional ML models show effective results in classification and regression tasks.
