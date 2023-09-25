import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample data
data = {'X1': [1, 2, 3, 4, 5],
        'X2': [2, 3, 4, 5, 6],
        'Y': [0, 1, 0, 1, 1]}

df = pd.DataFrame(data)

# Split data into independent variables (X) and the target variable (Y)
X = df[['X1', 'X2']]
Y = df['Y']

# Split the data into a training set and a test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Create a logistic regression model
model = LogisticRegression()

# Fit the model to the training data
model.fit(X_train, Y_train)

# Make predictions on the test data
Y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(Y_test, Y_pred)
confusion = confusion_matrix(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", report)
