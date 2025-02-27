import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import cv2
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, classification_report
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
val_dir = "D:/transfer_learning_medical_image_analysis/valid"
test_dir = "D:/transfer_learning_medical_image_analysis/test"

# Load Data
X_train, y_train = load_dataset(train_dir)
X_val, y_val = load_dataset(val_dir)
X_test, y_test = load_dataset(test_dir)

# Convert Regression to Classification (Categorizing Angles)
def categorize_angles(y):
    """Convert angle values into three categories: Low (0-30), Medium (30-60), High (60+)."""
    return np.digitize(y, bins=[30, 60])  # Bins: 0-30 (Class 0), 30-60 (Class 1), 60+ (Class 2)

y_train_cat = categorize_angles(y_train)
y_val_cat = categorize_angles(y_val)
y_test_cat = categorize_angles(y_test)

# Step 3: Build Model
def build_model(input_shape=(224, 224, 3), num_classes=3):
    """Builds a ResNet50-based classification model."""
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model layers
    
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Multi-class classification
    ])
    
    return model

model = build_model()

# Step 4: Compile Model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])  # Classification Task

# Step 5: Train Model
history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_val, y_val_cat),
    batch_size=32,
    epochs=10
)

# Step 6: Evaluate Model
test_loss, test_accuracy = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Step 7: Predictions & Metrics
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class labels

precision = precision_score(y_test_cat, y_pred, average='macro')
recall = recall_score(y_test_cat, y_pred, average='macro')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:\n", classification_report(y_test_cat, y_pred))

# Step 8: Visualizing Precision & Recall
plt.figure(figsize=(12, 6))

# Plot Precision & Recall
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Step 9: Save Model
model.save("neck_angle_classification_model.h5")
