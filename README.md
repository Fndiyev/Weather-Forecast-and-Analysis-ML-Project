# 🌦️ Weather Prediction and Analysis with Machine Learning

This project presents a comprehensive pipeline for analyzing weather data using machine learning and deep learning models. It includes data preprocessing, exploratory analysis, classification using traditional ML algorithms, clustering techniques, and a neural network for rainfall prediction.

## 🔍 Project Overview

This repository contains these main components:

1. **Web Scraping**
   Scrapes the Weather data from timeanddate site for implementing several ML applications.
   
2. **Classification**  
   Implements supervised learning models (Logistic Regression, SVM, KNN) to classify weather conditions based on meteorological data.

3. **Clustering**  
   Applies unsupervised learning (KMeans and DBSCAN) to discover hidden patterns in weather observations.

4. **Neural Network**  
   A PyTorch-based binary classifier trained to predict whether it will rain or not, using weighted loss to handle class imbalance.
   
5. **Time Series Analysis and Forecasting**
   Applies Arima/Sarima for temperature forecasting on 3 year hourly data of 188 cities.

6. **Regression**
   Implements supervised learning models (Linear Regression, Random Forest) to predict the continous weather conditions.
   
## 📁 Files

- `Final_project_Classification.ipynb`: Weather classification using multiple ML models.
- `Final_pr_Clustering.ipynb`: Exploratory clustering analysis.
- `Final_prj_neural_network.py`: Rain prediction with a neural network trained in PyTorch.
- `Final project.ipynb`: Time Series analysis and weather regression using multiple ML models.

## 🛠️ Tools & Libraries

- Python, Pandas, NumPy, Seaborn, Matplotlib
- Scikit-learn
- PyTorch
- Google Colab & Google Drive (for dataset access)

## 📊 Dataset

The project uses a historical weather dataset stored in Google Drive (`weather_2.csv`). It includes features like temperature, humidity, wind speed, visibility, UV index, and precipitation.

## 📈 Model Performance

The neural network achieves reasonable accuracy in predicting rainfall events, despite class imbalance. Weighted loss and normalization were applied to improve performance.
