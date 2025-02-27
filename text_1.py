import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np
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

# Step 2: Load the Trained Model
model = load_model("neck_angle_model_with_accuracy.h5")

# Step 3: Predict Function
def predict_angle(image_path):
    """Predicts the angle of the neck from the image."""
    # Preprocess the image
    image = preprocess_image(image_path)
    # Expand dimensions to match model input (batch size of 1)
    image = np.expand_dims(image, axis=0)
    
    # Predict the angle
    predicted_angle = model.predict(image)
    
    return predicted_angle[0][0]  # Return the predicted angle (scalar value)

# Step 4: Test with a Single Image
image_path = "D:/transfer_learning_medical_image_analysis/test/image_0062.jpg"  # Replace with your test image path

# Predict the angle for the test image
predicted_angle = predict_angle(image_path)
print(f"Predicted Angle: {360 - predicted_angle:.4f}")

# Step 5: Display the Image and the Predicted Angle
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the image and predicted angle
plt.imshow(image)
plt.title(f"Predicted Angle: {360 - predicted_angle:.4f}°")
plt.axis('off')  # Hide axes
plt.show()