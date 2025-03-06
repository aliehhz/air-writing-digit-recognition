import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os

def create_and_train_model():
    # Load MNIST dataset
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Reshape for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Build CNN model
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')  # 10 classes (0-9)
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Save the model

    path = os.path.join('artifacts', 'model')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path) 



if __name__ == "__main__":
    create_and_train_model()
