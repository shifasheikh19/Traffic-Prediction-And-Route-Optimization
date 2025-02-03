

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



merged_data = pd.read_csv('merged_data.csv')


print("Initial statistics:")
print(merged_data.describe())

merged_data.ffill(inplace=True)

print("\nStatistics after filling missing values:")
print(merged_data.describe())


numeric_cols = ['temp', 'feels_like', 'freeFlowSpeed', 'currentSpeed']
for col in numeric_cols:
    q_low = merged_data[col].quantile(0.01)
    q_high = merged_data[col].quantile(0.99)
    merged_data = merged_data[(merged_data[col] >= q_low) & (merged_data[col] <= q_high)]


print("\nStatistics after handling outliers:")
print(merged_data.describe())
merged_data = pd.read_csv('merged_data.csv')


numeric_cols = ['temp', 'feels_like', 'freeFlowSpeed', 'currentSpeed']


plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(y=merged_data[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 2, i)
    sns.histplot(merged_data[col], kde=True)
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()


merged_data['weather_main'] = merged_data['weather_main'].astype('category')


merged_data = pd.get_dummies(merged_data, columns=['weather_main'])


merged_data['temp_freeFlowSpeed'] = merged_data['temp'] * merged_data['freeFlowSpeed']
merged_data['feels_like_freeFlowSpeed'] = merged_data['feels_like'] * merged_data['freeFlowSpeed']

features = ['freeFlowSpeed', 'temp', 'feels_like', 'temp_freeFlowSpeed', 'feels_like_freeFlowSpeed'] + [col for col in merged_data.columns if col.startswith('weather_main_')]


target = 'currentSpeed'


merged_data.to_csv('cleaned_merged_data_fixed.csv', index=False)

print("Data cleaning and feature engineering completed successfully.")
