## Uses Camera connected to the Jetson Nano, takes an frame from the video and returns it as a 224x224x3 np array
import cv2
import time
from matplotlib import pyplot as plt

GSTREAMER_PIPELINE ="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

def captureFrame():

    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    for i in range(30):
        temp = cap.read()

    if cap.isOpened():
        ret_val, img = cap.read()
        img = cv2.resize(img, (224, 224))
       # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        print("Unable to open camera")
    
   # plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
   # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
   # plt.show()     
   # print("image type: ", type(img))
   # print("shape of image: ", img.shape)
    cv2.imwrite('./image.png',img)

    return img


if __name__ == "__main__":
    captureFrame()
