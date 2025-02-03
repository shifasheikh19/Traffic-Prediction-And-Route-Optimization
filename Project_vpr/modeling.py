from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt


merged_data = pd.read_csv('cleaned_merged_data_fixed.csv')


features = ['freeFlowSpeed', 'temp', 'feels_like', 'temp_freeFlowSpeed', 'feels_like_freeFlowSpeed'] + [col for col in merged_data.columns if col.startswith('weather_main_')]


target = 'currentSpeed'

X = merged_data[features]
y = merged_data[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = GradientBoostingRegressor()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)


print(f"Gradient Boosting - MSE: {mse}, MAE: {mae}")
print(f"Gradient Boosting - Actual Values: {y_test.values}")
print(f"Gradient Boosting - Predictions: {y_pred}")
