import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
dates = pd.date_range(start='2025-08-01', periods=30)
rainfall = np.random.normal(loc=50, scale=15, size=30).clip(min=0)
usage = np.random.normal(loc=1000, scale=250, size=30).clip(min=600)

df = pd.DataFrame({
    'Date': dates,
    'Rainfall_mm': rainfall,
    'Water_Usage_Liters': usage
})
df.to_csv('simulated_data.csv', index=False)

X = df[['Rainfall_mm']]
y = df['Water_Usage_Liters']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

df['Predicted_Usage'] = model.predict(X)

plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Water_Usage_Liters'], label='Actual Usage', color='green', marker='o')
plt.plot(df['Date'], df['Predicted_Usage'], label='Predicted Usage', color='blue', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Water Usage (L)')
plt.title('Actual vs Predicted Water Usage')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()