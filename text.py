import os
import json
import cv2
import numpy as np

# Paths to the folders containing JSON files and images
json_folder = "D:/transfer_learning_medical_image_analysis/val[1]/val\labels"
image_folder = "D:/transfer_learning_medical_image_analysis/val[1]/val\images"
output_folder = "D:/transfer_learning_medical_image_analysis/train[2]/train/annotated"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

def draw_annotations(image, annotations):
    for shape in annotations['shapes']:
        label = shape['label']
        points = shape['points']
        shape_type = shape['shape_type']

        if shape_type == 'rectangle':
            # Draw rectangles
            pt1 = tuple(map(int, points[0]))
            pt2 = tuple(map(int, points[1]))
            cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(image, label, (pt1[0], pt1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        elif shape_type == 'point':
            # Draw points
            pt = tuple(map(int, points[0]))
            cv2.circle(image, pt, 5, (0, 0, 255), -1)
            cv2.putText(image, label, (pt[0] + 5, pt[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    return image

# Iterate through all JSON files in the JSON folder
for json_file in os.listdir(json_folder):
    if json_file.endswith('.json'):
        json_path = os.path.join(json_folder, json_file)

        # Read the JSON file
        with open(json_path, 'r') as f:
            annotations = json.load(f)

        # Extract the image filename from the JSON
        image_filename = annotations['imagePath']
        image_path = os.path.join(image_folder, image_filename)

        # Check if the corresponding image exists
        if os.path.exists(image_path):
            # Read the image
            image = cv2.imread(image_path)

            # Draw annotations on the image
            annotated_image = draw_annotations(image, annotations)

            # Save the annotated image to the output folder
            output_path = os.path.join(output_folder, image_filename)
            cv2.imwrite(output_path, annotated_image)

            print(f"Annotated image saved: {output_path}")
        else:
            print(f"Image not found for JSON: {json_path}")

print("Annotation process completed.")
