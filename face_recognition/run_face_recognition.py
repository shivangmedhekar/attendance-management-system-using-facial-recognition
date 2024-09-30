import os
import subprocess
import mysql.connector

"""
Face Recognition Pipeline Script

This script orchestrates the process of checking database connectivity, 
training a face recognition model, and running real-time face recognition. 
It follows these steps:
1. Checks if a connection to the MySQL database can be established.
2. Checks if a pre-trained face recognition model exists.
3. If no trained model is found, it runs the face trainer script to train the model.
4. Finally, it runs the face recognition script for real-time recognition.

Usage:
1. Ensure the MySQL database is running and accessible.
2. Run this script to manage the face recognition workflow.
"""

# Function to check if a connection to the MySQL database can be established
def check_database_connection():
    """
    Check if the MySQL database can be connected to and query data.
    Returns True if successful, False otherwise.
    """
    try:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            passwd=""
        )
        mycursor = mydb.cursor()
        mycursor.execute("use testproj")  # Select the database
        mycursor.execute("SELECT * FROM test1")  # Query data from the table
        myresult = mycursor.fetchall()  # Fetch all results
        print("\n[INFO] Database connected successfully.")
        for x in myresult:
            print(x)  # Print each result
        return True
    except mysql.connector.Error as err:
        print(f"[ERROR] Failed to connect to database: {err}")
        return False

# Function to check if the face recognition model has been trained
def check_trained_model():
    """
    Check if the face recognition model has been trained.
    Returns True if the model exists, False otherwise.
    """
    return os.path.exists('face_recognition/trainer/trainer.yml')  # Check for the existence of the model file

# Function to run the face trainer script
def run_face_trainer():
    """
    Run the face trainer script to train the model.
    """
    print("\n[INFO] Starting face training...")
    subprocess.run(['python', 'face_recognition/scripts/face_trainer.py'], check=True)  # Execute the trainer script
    print("\n[INFO] Face training completed.")

# Function to run the face recognition script
def run_face_recognition():
    """
    Run the face recognition script for real-time recognition.
    """
    print("\n[INFO] Starting real-time face recognition...")
    subprocess.run(['python', 'face_recognition/scripts/face_recognition.py'], check=True)  # Execute the recognition script

# Main function to orchestrate the face training and recognition workflow
def main():
    """
    Main function to orchestrate face training and real-time recognition.
    """
    # Check database connection; exit if it fails
    if not check_database_connection():
        print("[ERROR] Unable to proceed without a database connection.")
        return

    # Check if the trained model exists; train if it doesn't
    if not check_trained_model():
        print("[INFO] No trained model found. Training the face recognition model first.")
        run_face_trainer()
    else:
        print("[INFO] Trained model already exists. Proceeding with face recognition.")

    # Run the face recognition script
    run_face_recognition()

# Entry point of the script
if __name__ == "__main__":
    main()  # Execute the main function
