import matplotlib.pyplot as plt
import seaborn as sns


y_test = [50, 45, 60, 55, 48]  # Replace with your actual data
y_pred = [52, 44, 59, 56, 47]  # Replace with your model predictions


import pandas as pd
data = pd.DataFrame({
    'Actual Speed': y_test,
    'Predicted Speed': y_pred
})


plt.figure(figsize=(10, 6))
sns.scatterplot(x='Actual Speed', y='Predicted Speed', data=data, color='blue', label='Predicted')
sns.lineplot(x='Actual Speed', y='Actual Speed', data=data, color='red', label='Ideal')
plt.title('Predicted vs Actual Traffic Speeds')
plt.xlabel('Actual Traffic Speed (km/h)')
plt.ylabel('Predicted Traffic Speed (km/h)')
plt.legend()
plt.grid(True)
plt.show()
