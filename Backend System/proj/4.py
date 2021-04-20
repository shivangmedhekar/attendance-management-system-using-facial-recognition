''''
Real Time Face Recogition
	==> Each face stored on dataset/ dir, should have a unique numeric integer ID as 1, 2, 3, etc                       
	==> LBPH computed model (trained faces) should be on trainer/ dir
Based on original code by Anirban Kar: https://github.com/thecodacus/Face-Recognition    

Developed by Marcelo Rovai - MJRoBot.org @ 21Feb18  

'''
import mysql.connector
import cv2
import numpy as np
import os 

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd=""
)
mycursor = mydb.cursor()
mycursor.execute("use testproj")


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'sohum', 'aditya', 'shivang']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img =cam.read()


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )
       
    count_of_insert = 1

    for(x,y,w,h) in faces:

        

        if count_of_insert == 0:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,0), 2)

        else:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,255), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            sql = "SELECT * FROM test1 where fname=%s"
            val = (id,)
            mycursor.execute(sql, val)
            myresult = mycursor.fetchall()

            student = []

            for tuple_db in myresult:
                for column in tuple_db:
                    student.append(column)

            # print(student)

            id_db = student[0]
            fname_db=student[1]
            lname_db=student[2]
            # print(id_db)
            # print(fname_db)
            # print(lname_db)

            sql = "SELECT COUNT(*) FROM test2 where fname=%s"
            val = (id,)
            
            mycursor.execute(sql, val)
            myresult = mycursor.fetchall()

            for tuple_db in myresult:
                for column in tuple_db:
                    count_of_insert = column
            

            

            if count_of_insert == 0:
                sql = "INSERT INTO test2 VALUES (%s, %s,%s)"
                val = (id_db,fname_db,lname_db)
                mycursor.execute(sql, val)

            mydb.commit()
            student.clear()

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        
        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
