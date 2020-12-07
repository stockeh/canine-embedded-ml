import tensorflow as tf
import numpy as np
import argparse
import collections
import time
import cv2
from matplotlib import pyplot as plt
from adafruit_servokit import ServoKit

from canine.jetson import capture
from canine.jetson import actions

MAX_BUFFER_LEN = 20
BUFFER = np.array([-1]*MAX_BUFFER_LEN)
LAST_ACTION = None
GSTREAMER_PIPELINE ="nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv flip-method=2 ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"

kit = ServoKit(channels=16)
kit.servo[0].angle = 100

def reward():
    kit.servo[0].angle = 60
    time.sleep(.10)
    kit.servo[0].angle = 100
    time.sleep(10) # enough time to eat treat

def make_decision():
    global LAST_ACTION
    labels, counts = np.unique(BUFFER, return_counts=True)
    counts = np.true_divide(counts, MAX_BUFFER_LEN)
    max_i = np.argmax(counts)
    if counts[max_i] >= 0.60:
        action = labels[max_i]

        if action != -1 and action != LAST_ACTION:
            if action == 0: # lying
                pass
            elif action == 1: # sitting
                reward()
            elif action == 2: # standing
                pass
            else:
                print(f'ERROR! Invalid action {action}')
                return
            
            LAST_ACTION = action

            print(f'{labels[max_i]} has majority with {counts[max_i]*100:.3f}%')
            print(BUFFER)


def make_inference(interpreter, X):
    X = tf.convert_to_tensor(X, dtype_hint=tf.float32)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        X = (X / input_scale + input_zero_point).astype(input_details['dtype'])

    interpreter.set_tensor(input_details['index'], X)

    start_t = time.time()
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    end_t = time.time()
    total_t = end_t - start_t
    
    action = np.argmax(output_data)

    # action must have min probability of class
    if output_data[0][action] < .40:
        action = -1
        print("unknown", output_data)

    if action == 0:
        print("lying", output_data)
    elif action == 1:
        print("sitting", output_data)
    elif action == 2:
        print("standing", output_data)

    return action, output_data


def main(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.resize_tensor_input(0, (1, 224, 224, 3), strict=0)
    interpreter.allocate_tensors()
    cap = cv2.VideoCapture(GSTREAMER_PIPELINE, cv2.CAP_GSTREAMER)
    
    for l in range (30):
        temp = cap.read()

    print('INFO: Finished initializing')

    i = 0
    while(True):

        if cap.isOpened():
            ret_val, img = cap.read()
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1,224,224,3)

            # X = tf.keras.applications.mobilenet.preprocess_input(X)
            Y, probs  = make_inference(interpreter, img)
            if Y != -1:
                BUFFER[i] = Y
                make_decision()

            i = (i+1) % MAX_BUFFER_LEN


if __name__ == "__main__":
    """
    Usage: python3 -u -m canine.jetson.main -m /home/nano/Development/canine/models/MobileNetV2-Dog.tflite
    """

    gpus = tf.config.get_visible_devices('GPU')
    for device in gpus:
        print(device)
        tf.config.experimental.set_memory_growth(device, True)

    parser = argparse.ArgumentParser(description='neural network model')
    parser.add_argument('-m', '--model', metavar='model', type=str,
                        required=True, help='the path to model file')
    args = parser.parse_args()
    main(tflite_path=args.model)
