import os
import zipfile
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, 
                                     BatchNormalization, Dropout, GlobalAveragePooling2D)

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

train_dir = '/kaggle/input/ee-dataset/Train/Train'
test_dir = '/kaggle/input/ee-dataset/Test/Test'

print("Train Klasörü:", train_dir)
print("Test Klasörü:", test_dir)

data_ayr = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,# validation set %20 training set %80
    horizontal_flip=True,
    zoom_range=0.2
)

train_generator = data_ayr.flow_from_directory(
    directory=train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = data_ayr.flow_from_directory(
    directory=train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)


test_dataAyr = ImageDataGenerator(rescale=1./255)
test_generator = test_dataAyr.flow_from_directory(
    directory=test_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
print(train_generator.class_indices)
print(train_generator.samples)        

EEmodel = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),  
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2), 
    Dense(len(train_generator.class_indices), activation='softmax')
])

EEmodel.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

EEmodel.summary() 

history = EEmodel.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    verbose=1
)

test_loss, test_acc = EEmodel.evaluate(test_generator, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

test_generator.reset()
predictions = EEmodel.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

cm = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:\n", cm)

print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
