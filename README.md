# Bike-Sharing Demand Prediction

## Overview

Bike-sharing systems have become a popular mode of transportation in urban areas, providing an eco-friendly and efficient way to travel short distances. However, managing the supply and demand of bikes is a significant challenge for operators. Accurate prediction of bike-sharing demand can help in optimizing bike distribution, reducing operational costs, and enhancing user experience.

## Objective

The goal of this project is to develop a deep learning model that predicts the bike-sharing demand for the next 60 minutes for each station based on historical data. Students will work individually to analyze the data, build and evaluate models, and provide actionable business recommendations.

## Key Business Questions

- What are the top 5 stations with the highest demand over a given period?
- Which stations experience the lowest demand?
- What are the demand trends over time for different stations?
- How does bike demand vary across different times of the day?
- What are the ride demand patterns across different stations?

## Methodology

1. **Data Analysis**:
   - Exploratory Data Analysis (EDA) was conducted to answer the key business questions.
   
2. **Model Development**:
   Three deep learning models were developed:
   - **LSTM-based Convolutional Model**: Used to capture both spatial and temporal patterns.
   - **LSTM with Attention Mechanism**: Enhanced model to attend to spatial features.
   - **Graph Convolutional Network (GCN)**: Represented stations as nodes and rides as edges to understand inter-station relationships.

3. **Evaluation**:
   The models were evaluated using **Mean Absolute Error (MAE)** as the primary metric, and optimization was performed using the **ADAM optimizer**.

## Files and Directories


- `data/`: Contains the historical data files for training the models.
- `model/`: Contains the saved models and scripts for model building and evaluation.
- `notebooks/`: Jupyter notebooks for the analysis and model training.



## Requirements

- Python 3.x
- Libraries: `numpy`, `pandas`, `tensorflow`, `keras`, `matplotlib`, `sklearn`

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/AliMehizel/bike_sharing_prediction__.git
