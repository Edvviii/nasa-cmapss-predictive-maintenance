AI-Based Predictive Maintenance using NASA C-MAPSS Dataset
Project Overview

This project implements a machine learning-based predictive maintenance system to estimate the Remaining Useful Life (RUL) of turbofan engines using the NASA C-MAPSS dataset. The model analyzes multivariate sensor data from aircraft engines and predicts how many operational cycles remain before failure.

Predictive maintenance helps industries reduce downtime, prevent unexpected failures, and optimize maintenance scheduling.

Dataset

The project uses the NASA Turbofan Engine Degradation Simulation Dataset (C-MAPSS) provided by NASA’s Prognostics Center of Excellence.

Dataset Source:
https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

The dataset includes:

Multiple turbofan engines

Time-series sensor measurements

Operational settings

Engine degradation until failure

Files Used

Training datasets:

train_FD001.txt

train_FD002.txt

train_FD003.txt

train_FD004.txt

Testing datasets:

test_FD001.txt

test_FD002.txt

test_FD003.txt

test_FD004.txt

Remaining Useful Life labels:

RUL_FD001.txt

RUL_FD002.txt

RUL_FD003.txt

RUL_FD004.txt

Problem Statement

Given historical engine sensor data, predict the Remaining Useful Life (RUL) of each engine.

This is formulated as a supervised regression problem where:

Input:

Engine operational settings

21 sensor measurements

Time cycles

Output:

Remaining useful life of the engine

Machine Learning Model

This project uses a Random Forest Regressor for RUL prediction.

Why Random Forest?

Random Forest was selected because it:

Handles high-dimensional data well

Captures nonlinear relationships

Is robust to noise in sensor data

Requires minimal feature engineering

Provides feature importance insights

Project Pipeline

The system follows these steps:

Data Loading
Load all training and testing datasets.

Data Preprocessing

Assign column names

Remove empty columns

Merge engine cycles

Compute Remaining Useful Life (RUL)

Feature Preparation
Use sensor measurements and operational settings as input features.

Feature Scaling
Normalize features using StandardScaler.

Model Training
Train a Random Forest Regression model on the training datasets.

Model Evaluation
Evaluate predictions using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

Visualization
Generate plots for:

Actual vs Predicted RUL

Feature Importance

Prediction Error Distribution

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/nasa-cmapss-predictive-maintenance.git

Move into the project folder:

cd nasa-cmapss-predictive-maintenance

Install required dependencies:

pip install -r requirements.txt
Running the Project

Run the main training script:

python src/train_random_forest.py

The script will:

Train the model

Predict RUL on test datasets

Print evaluation metrics

Generate visualization plots

Evaluation Metrics

The model performance is evaluated using:

RMSE – Root Mean Squared Error
MAE – Mean Absolute Error
R² Score – Model goodness of fit

These metrics measure how accurately the model predicts the remaining engine life.

Visualizations

The project generates the following visualizations:

Actual vs Predicted RUL

Feature Importance Plot

Prediction Error Distribution

These plots help understand model behavior and prediction accuracy.

Technologies Used

Python

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

Applications

Predictive maintenance models like this are widely used in:

Aerospace industry

Manufacturing

Industrial equipment monitoring

IoT-based maintenance systems

Smart factories

Future Improvements

Possible improvements include:

Implementing Deep Learning models (LSTM / GRU)

Using time-series windowing techniques

Applying advanced feature engineering

Building real-time predictive maintenance dashboards

Deploying the model using MLOps pipelines

Author

Edwin Wilson
B.Tech Computer Science Engineering
