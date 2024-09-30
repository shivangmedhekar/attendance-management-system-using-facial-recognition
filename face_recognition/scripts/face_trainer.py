import cv2
import numpy as np
from PIL import Image
import os

"""
Face Training Script

This script trains a face recognition model using images stored in a specified dataset directory. 
It utilizes OpenCV's LBPH (Local Binary Patterns Histograms) face recognizer to create a model 
that can recognize faces based on the images captured. The trained model is saved to a specified 
trainer directory for later use in face recognition.

Usage:
1. Ensure that face images are stored in the 'face_recognition/dataset' directory, 
   with each image named in the format: User.{face_id}.{count}.jpg (e.g., User.1.1.jpg).
2. Run the script to train the model on the available images.
3. The trained model will be saved in the 'face_recognition/trainer' directory as 'trainer.yml'.
"""

# Function to train the face recognition model
def train_faces():
    # Directory containing face images
    path = 'face_recognition/dataset'
    
    # Create an LBPH face recognizer instance
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load the pre-trained face detector from OpenCV
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Function to retrieve images and corresponding labels from the dataset
    def get_images_and_labels(path):
        # Get a list of all image file paths in the dataset directory
        image_paths = [os.path.join(path, f) for f in os.listdir(path)]
        face_samples = []  # List to hold face samples
        ids = []           # List to hold corresponding face IDs

        # Iterate through each image path to process the images
        for image_path in image_paths:
            # Open the image, convert it to grayscale
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')  # Convert to NumPy array
            
            # Extract the face ID from the image filename
            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)  # Detect faces in the image

            # Extract each face from the detected faces
            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y + h, x:x + w])  # Append face sample
                ids.append(id)  # Append corresponding ID

        return face_samples, ids  # Return face samples and IDs

    print("\n[INFO] Training faces...")
    faces, ids = get_images_and_labels(path)  # Retrieve images and labels
    recognizer.train(faces, np.array(ids))  # Train the recognizer with the face samples and IDs

    # Create the trainer directory if it doesn't exist
    os.makedirs('face_recognition/trainer', exist_ok=True)
    # Save the trained model to a YAML file
    recognizer.write('face_recognition/trainer/trainer.yml')
    print(f"\n[INFO] {len(np.unique(ids))} faces trained.")  # Print the number of unique faces trained

# Main script entry point
if __name__ == "__main__":
    train_faces()  # Call the function to train faces
