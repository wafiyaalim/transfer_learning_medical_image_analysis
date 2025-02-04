import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Enable eager execution for debugging
tf.config.run_functions_eagerly(True)

# Step 1: Preprocessing Function
def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocesses an image: loads, resizes, and normalizes."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size) / 255.0
    return image

# Step 2: Load Dataset
def load_dataset(data_dir, target_size=(224, 224)):
    """
    Loads dataset from a flat directory structure where filenames contain annotations.
    Assumes filenames are of the format "image_30.jpg" (angle_30).
    """
    X = []
    y = []

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)

        if os.path.isfile(item_path):  # Handle image files
            try:
                # Extract label (angle) from filename (e.g., "image_30.jpg")
                label = float(item.split('_')[1].split('.')[0])  # Extract label from filename
                X.append(preprocess_image(item_path, target_size))
                y.append(label)
            except (IndexError, ValueError):
                print(f"Skipping file {item}: Unable to parse label")
    
    return np.array(X), np.array(y)

# Paths
train_dir = "D:/transfer_learning_medical_image_analysis/train"
val_dir ="D:/transfer_learning_medical_image_analysis/valid"
test_dir = "D:/transfer_learning_medical_image_analysis/test"

# Load Data
X_train, y_train = load_dataset(train_dir)
X_val, y_val = load_dataset(val_dir)
X_test, y_test = load_dataset(test_dir)

# Step 3: Data Augmentation Pipeline
augmentation_layers = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.1),  # Rotate up to 10%
    RandomZoom(0.2)       # Zoom by up to 20%
])

# Step 4: Build Model
def build_model(input_shape=(224, 224, 3)):
    """Builds a ResNet50-based regression model."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers

    model = Sequential([
        augmentation_layers,  # Add data augmentation as the first layer
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear')  # Regression for angle prediction
    ])

    return model

model = build_model()

# Step 5: Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae'])  # Mean Absolute Error (MAE) as an additional metric

# Step 6: Train Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=10
)

# Step 7: Evaluate Model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Mean Absolute Error: {test_mae:.4f}")

# Additional Metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Step 8: Visualize Training History
plt.figure(figsize=(12, 6))

# Plotting Loss
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting MAE
plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Mean Absolute Error Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

# Plotting Predicted vs True Values
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title('Predicted vs True Values')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')

plt.tight_layout()
plt.show()

# Step 9: Save Model
model.save("neck_angle_model_with_augmentations.h5")
