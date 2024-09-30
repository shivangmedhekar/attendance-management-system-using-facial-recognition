import cv2
import os

"""
Face Capture Script

This script captures face images from the webcam for a specified user ID. 
It uses OpenCV to detect faces in real-time and saves the captured images 
to a specified directory in grayscale format. The script allows the user 
to capture up to 50 images per face and displays the webcam feed with 
detected faces highlighted.

Usage:
1. Run the script.
2. Enter a numeric user ID when prompted.
3. Look at the camera to capture your face images.
4. Press 'ESC' to exit early or wait until 50 images are captured.
"""

# Function to capture faces for a given user ID
def capture_faces(face_id):
    # Initialize video capture using the default webcam (device 0)
    cam = cv2.VideoCapture(0)
    # Set video resolution width and height
    cam.set(3, 640)  # Width
    cam.set(4, 480)  # Height

    # Load the pre-trained face detector from OpenCV
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Directory to save the captured face images
    dataset_dir = "face_recognition/dataset"
    # Create the directory if it doesn't exist
    os.makedirs(dataset_dir, exist_ok=True)

    # Print instruction message for the user
    print("\n[INFO] Initializing face capture. Look at the camera...")

    # Counter for the number of face images captured
    count = 0

    # Start capturing video frames in a loop
    while True:
        # Capture frame-by-frame
        ret, img = cam.read()
        
        # Convert the frame to grayscale (required for face detection)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale frame
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        # Iterate through the detected faces and process each one
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face in the video frame
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Increment the face count
            count += 1

            # Save the captured face image as a file in the dataset directory
            cv2.imwrite(f"{dataset_dir}/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

            # Display the video frame with the drawn rectangle
            cv2.imshow('image', img)

        # Check for 'ESC' key (ASCII 27) to exit the loop, or stop after capturing 50 images
        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 50:
            break

    # Clean up and release resources after capturing
    print("\n[INFO] Exiting and cleaning up...")
    cam.release()
    cv2.destroyAllWindows()

# Main script entry point
if __name__ == "__main__":
    # Ask the user for a numeric user ID to assign to the captured face images
    user_id = input("\nEnter user ID and press Enter: ")
    capture_faces(user_id)
