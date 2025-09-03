import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
dates = pd.date_range(start='2025-08-01', periods=30)
rainfall_mm = np.random.normal(loc=50, scale=20, size=30).clip(min=0)
usage_liters = np.random.normal(loc=1000, scale=300, size=30).clip(min=500)

df = pd.DataFrame({
    'Date': dates,
    'Rainfall_mm': rainfall_mm,
    'Water_Usage_Liters': usage_liters
})

df.to_csv('simulated_data.csv', index=False)

X = df[['Rainfall_mm']]
y = df['Water_Usage_Liters']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

df['Predicted_Usage'] = model.predict(X)

def classify_usage(row):
    if row['Predicted_Usage'] > 1200 and row['Rainfall_mm'] < 30:
        return "Alert: High usage expected, low rainfall"
    elif row['Predicted_Usage'] < 800:
        return "Usage within safe range"
    else:
        return "Monitor usage"

df['Decision'] = df.apply(classify_usage, axis=1)

print(df[['Date', 'Rainfall_mm', 'Predicted_Usage', 'Decision']].head())

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