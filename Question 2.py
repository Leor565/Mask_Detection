# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 23:55:47 2024

@author: leor7
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set dataset path
data_path = r"C:\Users\leor7\OneDrive\Documents\assignment 3 nn\Dataset"
categories = ['with_mask', 'without_mask']

# Prepare data
data = []
labels = []

for category in categories:
    folder = os.path.join(data_path, category)
    label = categories.index(category)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_array = tf.keras.preprocessing.image.load_img(img_path, target_size=(100, 100))
        img_array = tf.keras.preprocessing.image.img_to_array(img_array) / 255.0
        data.append(img_array)
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)

# Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, batch_size=32)

# Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# --- Add Testing with a Single Image ---
def test_single_image(image_path, model):
    """
    Test a single image and predict its class (with_mask or without_mask).
    """
    # Load and preprocess the image
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(100, 100))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Predict using the trained model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions) 
    
    # Map the index to class labels
    class_labels = ['with_mask', 'without_mask']
    print(f"Prediction for the image '{os.path.basename(image_path)}': {class_labels[predicted_class]}")

# Test a single image
test_image_path = r"C:\Users\leor7\OneDrive\Documents\assignment 3 nn\Dataset\with_mask\0.jpg"  # Replace with the path to your test image
test_single_image(test_image_path, model)
