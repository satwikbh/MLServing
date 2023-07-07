import os

import tensorflow as tf
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values between 0 and 1
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Expand dimensions to represent grayscale images
x_train = tf.expand_dims(x_train, -1)
x_test = tf.expand_dims(x_test, -1)

# Convert labels to one-hot encoding
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = tf.keras.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ]
)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))

test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Define the path to save the model
model_path = f"{os.getcwd()}/MnistCNNModel.h5"

# Save the model to the specified path
model.save(model_path)

print("Model saved successfully.")
