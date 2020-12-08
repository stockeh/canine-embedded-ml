import time
from adafruit_servokit import ServoKit

def sit():
    kit = ServoKit(channels=16)
    print('Finished Init()')

    kit.servo[0].angle = 35
    time.sleep(1)
    kit.servo[0].angle = 150
    time.sleep(1)
    kit.servo[0].angle = 90
    time.sleep(1)
    kit.servo[0].angle = 35
