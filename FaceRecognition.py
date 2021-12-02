# OpenCV program to detect cat face in real time
# import libraries of python OpenCV
# where its functionality resides
import cv2
import numpy as np
from playsound import playsound

# load the required trained XML classifiers
# https://github.com/Itseez/opencv/blob/master/
# data/haarcascades/haarcascade_frontalcatface.xml
# Trained XML classifiers describes some features of some
# object we want to detect a cascade function is trained
# from a lot of positive(faces) and negative(non-faces)
# images.
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture frames from a camera
cap = cv2.VideoCapture(0)

# describe the type of font
    # to be used.
font = cv2.FONT_HERSHEY_SIMPLEX

# loop runs if capturing has been initialized.
while 1:

    # reads frames from a camera
    ret, img = cap.read()

    # convert to gray scale of each frames
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Detects faces of different sizes in the input image
    faces = face_cascade.detectMultiScale(gray, 1.3,10, 5)
    if len(faces) == 1:
        cv2.putText(img,'Get out stinky!(useSpray)',(0,20),font, .8,(0, 255, 255),2,cv2.LINE_4)
        #playSound
        playsound('nostinky.mp3')
        # activate bluetooth useSpray()

    for (x,y,w,h) in faces:
        # To draw a rectangle in a face
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
        cv2.putText(img,'Face',(x+w-100,y+h+25),font, 1,(0, 255, 255),2,cv2.LINE_4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]



    # Display an image in a window
    cv2.imshow('img',img)

    # Wait for Esc key to stop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()
