from gpiozero import MotionSensor, Button, LED
from picamera import PiCamera
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import numpy
import smtplib
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition
import imutils
import io
import os
import pickle
import datetime
import time
from time import sleep
import cv2
import RPi.GPIO as GPIO

from_email_addr = 'krishchudal9@gmail.com'
from_email_password = 'bqrnaqcezcttgdby'
to_email_addr = 'krishchudal9@gmail.com'

img_counter = 0
data_dir = 'dataset/'

RELAY = 17
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(RELAY, GPIO.OUT)
GPIO.output(RELAY, GPIO.LOW)
GPIO.setup(22, GPIO.OUT)

# Initialize 'currentname' to trigger only when a new person is identified.
currentname = "unknown"
# Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
# use this xml file
# https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
cascade = "haarcascade_frontalface_default.xml"

# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + face detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(PiCamera=True).start()
time.sleep(2.0)

# start the FPS counter
fps = FPS().start()

prevTime = 0
doorUnlock = False

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to 500px (to speedup processing)
    frame = vs.read()
    frame = imutils.resize(frame, width=500)

    # convert the input frame from (1) BGR to grayscale (for face
    # detection) and (2) from BGR to RGB (for face recognition)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect faces in the grayscale frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
                                      minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # OpenCV returns bounding box coordinates in (x, y, w, h) order
    # but we need them in (top, right, bottom, left) order, so we
    # need to do a bit of reordering
    boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

    # compute the facial embeddings for each face bounding box
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []

    # loop over the facial embeddings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(data["encodings"],
                                                 encoding)
        name = "Unknown"  # if face is not recognized, then print Unknown

        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # to unlock the door
            GPIO.output(RELAY, GPIO.HIGH)
            GPIO.output(22, GPIO.LOW)
            prevTime = time.time()
            doorUnlock = True
            print("door unlock")
            print("Alarm disabled")

        else:

            matches = face_recognition.compare_faces(
                data["encodings"], encoding)
            name = "Unknown"
            doorUnlock = False
            GPIO.output(RELAY, GPIO.LOW)
            GPIO.output(22, GPIO.HIGH)
            print("door lock")
            print("Alarm ON")
            
            for filename in os.listdir(data_dir):
                if filename.endswith('.jpg'):
                    counter = int(filename.split('_')[-1].split('.')[0])
                    img_counter = max(img_counter, counter)
                    
            # increment the image counter for the new image
            img_counter += 1

            # capture picture from PiCamera
            print("[INFO] Capturing image...")
            img_name = "dataset/" + "/alert_{}.jpg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))

            # read image file
            with open(img_name, 'rb') as f:
                img_data = f.read()

            msg = MIMEMultipart()
            msg['Subject'] = 'Alert...!!! Unknown person detected outside the door.'
            msg['From'] = from_email_addr
            msg['To'] = to_email_addr

            # attach image to message
            attachment = MIMEImage(img_data, name='alert.jpg')
            msg.attach(attachment)

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(from_email_addr, from_email_password)
            server.sendmail(from_email_addr, to_email_addr, msg.as_string())
            server.quit()

            print('Email sent')
            doorUnlock = False

            # loop over the matched indexes and maintain a count for
            # each recognized face face

    # lock the door after 5 seconds
    if doorUnlock == True and time.time() - prevTime > 5:
        doorUnlock = False
        GPIO.output(RELAY, GPIO.LOW)
        print("door lock")


# loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image - color is in BGR
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (255, 0, 0), 2)

    # display the image to our screen
    cv2.imshow("Facial Recognition is Running", frame)
    key = cv2.waitKey(1) & 0xFF

    # quit when 'q' key is pressed
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
