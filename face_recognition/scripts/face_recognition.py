import cv2
import numpy as np
import os

"""
Face Recognition Script

This script recognizes faces using a pre-trained face recognition model. 
It captures video from the webcam, detects faces in the video stream, 
and identifies them based on the trained model. The recognized faces are 
displayed on the video feed along with their confidence levels.

Usage:
1. Ensure that the pre-trained model is available in 'face_recognition/trainer/trainer.yml'.
2. Run the script to start the webcam feed and recognize faces in real-time.
3. Press the 'Esc' key to exit the program.
"""

# Function to recognize faces using the trained model
def recognize_faces():
    # Create an LBPH face recognizer instance and load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('face_recognition/trainer/trainer.yml')

    # Load the pre-trained face detector from OpenCV
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # List of known names corresponding to face IDs
    names = ['None', 'Sohum', 'Aditya', 'Shivang']
    
    # Initialize video capture from the webcam
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # Set video width
    cam.set(4, 480)  # Set video height

    # Calculate minimum width and height for face detection
    min_w = 0.1 * cam.get(3)
    min_h = 0.1 * cam.get(4)

    while True:
        # Capture frame-by-frame
        ret, img = cam.read()
        
        # Convert the captured frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(min_w), int(min_h)),
        )

        # Iterate through detected faces
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Recognize the face and get confidence level
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check confidence level and assign names accordingly
            if confidence < 100:
                name = names[id]
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "unknown"
                confidence_text = f"  {round(100 - confidence)}%"

            # Display the name and confidence on the video feed
            cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        # Show the captured frame with detected faces
        cv2.imshow('camera', img)

        # Break the loop if the 'Esc' key is pressed
        if cv2.waitKey(10) & 0xff == 27:
            break

    print("\n[INFO] Exiting and cleaning up...")
    cam.release()  # Release the camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

# Main script entry point
if __name__ == "__main__":
    recognize_faces()  # Call the function to recognize faces
