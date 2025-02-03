# Traffic-Prediction-And-Route-Optimization
This project uses machine learning for predicting traffic speeds and integrates the predictions into a vehicle routing problem (VRP) to optimize delivery routes. The primary objective is to showcase how real-time traffic conditions can influence logistical decisions and improve route planning efficiency.

**Requirements**

Python 3.x

pandas

numpy

scikit-learn

ortools

matplotlib

**Project Overview**

**Traffic Prediction:**

The project uses a Gradient Boosting Regressor model to predict traffic speeds based on weather and traffic-related features.
Features include freeFlowSpeed, temp, feels_like, and other weather-related parameters.
Data preprocessing steps involve imputation, scaling, and splitting into training and testing sets.

**Vehicle Routing Problem (VRP):**

After predicting traffic speeds, the project adjusts the distance matrix used for the VRP based on predicted traffic conditions.
The project uses Google's OR-Tools to solve the VRP and find the optimal route with adjusted costs based on traffic conditions.

**Visualization:**

A scatter plot compares actual and predicted traffic speeds.
A bar chart compares the total cost of the route before and after adjusting for traffic conditions.
