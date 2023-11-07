from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target variable

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Create a Gaussian Naive Bayes classifier
nb_classifier = GaussianNB()

# Step 4: Train the classifier on the training data
nb_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = nb_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Display a classification report
target_names = iris.target_names
print(classification_report(y_test, y_pred, target_names=target_names))
