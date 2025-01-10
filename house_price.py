# Import libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Dataset: House size (sq ft) and corresponding price (in $1000)
# X = np.array([[500], [1000], [1500], [2000], [2500]])  # Size
# y = np.array([50, 100, 150, 200, 250])  # Price

X = np.array([[1500, 3], [2000, 4], [1700, 3], [1800, 4]])
y = np.array([300000, 400000, 350000, 380000])

# Train the model
model = LinearRegression()
model.fit(X, y)

# Predict price for a new house
new_size = np.array([[1800 , 3]])
predicted_price = model.predict(new_size)
print(predicted_price)
print(f"Predicted price for 1800 sq ft house: ${predicted_price[0] * 1000:.2f}")
