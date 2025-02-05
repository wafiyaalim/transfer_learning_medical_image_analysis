import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)

# Function: Preprocess Image
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesses an image: loads, resizes, and normalizes."""
    try:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size) / 255.0
        return image
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Function: Load Dataset
def load_dataset(data_dir, target_size=(224, 224)):
    """
    Loads dataset from a directory where filenames contain annotations.
    Assumes filenames are in the format "image_30.jpg" (angle_30).
    """
    X, y = [], []
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isfile(item_path):
            try:
                label = float(item.split('_')[1].split('.')[0])  # Extract angle from filename
                image = preprocess_image(item_path, target_size)
                if image is not None:
                    X.append(image)
                    y.append(label)
            except (IndexError, ValueError):
                print(f"Skipping file {item}: Unable to parse label")

    return np.array(X), np.array(y)

# Paths
train_dir = "C:/Users/sahil/Documents/transfer_learning_medical_image_analysis/train"
val_dir = "C:/Users/sahil/Documents/transfer_learning_medical_image_analysis/valid"
test_dir = "C:/Users/sahil/Documents/transfer_learning_medical_image_analysis/test"

# Load Data
X_train, y_train = load_dataset(train_dir)
X_val, y_val = load_dataset(val_dir)
X_test, y_test = load_dataset(test_dir)

# Data Augmentation
augmentation_layers = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.1),
    RandomZoom(0.2)
])

# Function: Build EfficientNetB3 Model
def build_efficientnet_model(input_shape=(224, 224, 3)):
    """Builds an EfficientNetB3-based regression model."""
    base_model = EfficientNetB3(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers

    model = Sequential([
        augmentation_layers,  # Data augmentation
        base_model,
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Regression output
    ])
    
    return model

# Compile Model
model = build_efficientnet_model()
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='mean_squared_error',
              metrics=['mae'])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=32, epochs=15)

# Evaluate Model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")

# Predictions & Metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE: {mae:.4f}, R² Score: {r2:.4f}")

# Save Model
model.save("neck_angle_efficientnetB3.h5")

# Plot Training History
plt.figure(figsize=(12, 6))

# Loss Curve
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss Over Epochs')

# MAE Curve
plt.subplot(2, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.title('Mean Absolute Error (MAE) Over Epochs')

# Predicted vs True Values
plt.subplot(2, 2, 3)
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs True Values')

# Residual Plot (Errors)
residuals = y_test - y_pred
plt.subplot(2, 2, 4)
sns.histplot(residuals, bins=25, kde=True, color='red')
plt.axvline(x=0, color='black', linestyle='dashed')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Residual Distribution')

plt.tight_layout()
plt.show()