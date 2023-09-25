import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Sample data
data = {'X1': [1, 2, 3, 4, 5],
        'X2': [2, 3, 4, 5, 6],
        'Y': [2, 4, 5, 4, 5]}

df = pd.DataFrame(data)

# Split data into independent variables (X) and the dependent variable (Y)
X = df[['X1', 'X2']]
Y = df['Y']

# Ensure that you have at least two samples in both training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

if len(X_train) < 2 or len(X_test) < 2:
    print("Insufficient data points for R2 calculation.")
else:
    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the training data
    model.fit(X_train, Y_train)

    # Make predictions on the test data
    Y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)

    # Print the coefficients and intercept of the linear equation
    print("Coefficients:", model.coef_)
    print("Intercept:", model.intercept_)
