import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[500], [1000], [1500], [2000], [2500]])  # Size
y = np.array([50, 100, 150, 200, 250])  # Price

model = LinearRegression()
model.fit(X, y)

new_size = np.array([[1800]])
predicted_price = model.predict(new_size)

# f-string with debugging
print(f"Predicted price for {new_size[0][0]=}: ${predicted_price[0] * 1000:.2f}")
