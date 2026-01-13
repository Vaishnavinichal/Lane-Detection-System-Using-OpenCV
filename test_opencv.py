import cv2
import numpy as np
import sys

print("Python:", sys.version.splitlines()[0])
print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)

cap = cv2.VideoCapture(0)    # try webcam index 0
if not cap.isOpened():
    print("Could not open webcam (index 0). Try index 1 or close other apps using camera.")
else:
    ret, frame = cap.read()
    if ret:
        print("Captured frame shape:", frame.shape)
        cv2.imshow("Webcam Test - Press any key", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Opened webcam but couldn't read a frame.")
cap.release()
