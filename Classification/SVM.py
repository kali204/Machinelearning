from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load the Breast Cancer dataset
breast_cancer = datasets.load_breast_cancer()
X = breast_cancer.data  # Features
y = breast_cancer.target  # Target variable (0: malignant, 1: benign)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Step 4: Train the classifier on the training data
svm_classifier.fit(X_train, y_train)

# Step 5: Make predictions on the test data
y_pred = svm_classifier.predict(X_test)

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 7: Display a classification report
target_names = ['Malignant', 'Benign']
print(classification_report(y_test, y_pred, target_names=target_names))
