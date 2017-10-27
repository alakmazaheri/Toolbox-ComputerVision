"""
AVA LAKMAZAHERI
IMAGE PROCESSING TOOLBOX
Experiment with face detection and image filtering using OpenCV
"""

import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
kernel = np.ones((100, 100), 'uint8')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    faces = face_cascade.detectMultiScale(frame, scaleFactor=1.2, minSize=(20, 20))
    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w, :] = cv2.dilate(frame[y:y+h, x:x+w, :], kernel)
        cv2.ellipse(frame, (int(x+w/2), int(y+3*h/5)), (int(w/4),int(h/4)), 110, -90, 50, (0, 0, 0), 5)
        cv2.circle(frame, (int(x+w/4), int(y+h/3)), 20, (255,255,255), -1)
        cv2.circle(frame, (int(x+3*w/4), int(y+h/3)), 20, (255,255,255), -1)
        cv2.circle(frame, (int(x+w/4), int(y+h/3)), 5, (0,0,0), -1)
        cv2.circle(frame, (int(x+3*w/4), int(y+h/3)), 5, (0,0,0), -1)
        cv2.line(frame, (int(x+w/2), int(y+2*h/3)), (int(x+w/2+10), int(y+2*h/3)), (0,0,0), 2)
        cv2.line(frame, (int(x+w/2), int(y+2*h/3)), (int(x+w/2+10), int(y+h/2)), (0,0,0), 2)

        cv2.ellipse(frame, (int(x + w/4), int(y + h/4+5)), (int(w/4),int(h/4)), 120, 90, 180, (0, 0, 0), 2)
        cv2.ellipse(frame, (int(x + 3*w/4), int(y + h/4+5)), (int(w/4),int(h/4)), 120, 110, 200, (0, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
