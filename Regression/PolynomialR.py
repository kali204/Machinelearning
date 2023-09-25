import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 6])

# Reshape X into a 2D array (required for scikit-learn)
X = X.reshape(-1, 1)

# Create polynomial features (e.g., quadratic, cubic)
poly = PolynomialFeatures(degree=2)  # You can change the degree as needed
X_poly = poly.fit_transform(X)

# Create a polynomial regression model
model = LinearRegression()

# Fit the model to the polynomial features
model.fit(X_poly, Y)

# Predict using the model
Y_pred = model.predict(X_poly)

# Plot the original data and the polynomial regression line
plt.scatter(X, Y, color='blue', label='Actual Data')
plt.plot(X, Y_pred, color='red', label='Polynomial Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
