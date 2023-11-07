import tensorflow as tf
from tensorflow import keras

# Step 1: Import Libraries

# Step 2: Data Preparation
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to [0, 1]

# Step 3: Model Architecture
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Step 4: Loss and Optimizer
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Step 5: Training
model.fit(x_train, y_train, epochs=5)

# Step 6: Evaluation
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100}%")

# Step 7: Inference
predictions = model.predict(x_test[:5])  # Make predictions for the first 5 test images
print("Predictions:", predictions)
