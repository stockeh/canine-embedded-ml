# Uses Camera connected to the Jetson Nano, takes an frame from the video and returns it as a 224x224x3 np array

import numpy as np
import cv2

GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink wait-on-eos=false max-buffers=1 drop=True'


def captureFrame():
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
#    cap.set(3, 224)
#    cap.set(4, 224)

    while True and video_capture.isOpened():
        key, frame = cap.read()
        resize = cv2.resize(frame, (224, 224))
        
        if not return_key:
            break

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    frame = cv2.resize(frame, (224,224), interpolation=cv2.INTER_AREA)
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    
    return frame

if __name__ == "__main__":
    captureFrame()
