import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logging.basicConfig(level=logging.INFO)

X = np.array([[500], [1000], [1500], [2000], [2500]])
y = np.array([50, 100, 150, 200, 250])

model = LinearRegression()
model.fit(X, y)

new_size = np.array([[1800]])
predicted_price = model.predict(new_size)

logging.info(f"Training completed. Coefficients: {model.coef_}")
logging.info(f"Prediction for size {new_size[0][0]}: ${predicted_price[0] * 1000:.2f}")
