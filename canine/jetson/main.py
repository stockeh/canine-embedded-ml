import tensorflow as tf
import numpy as np
import argparse
import collections
import time

# from canine.jetson import capture
from canine.jetson import actions

MAX_BUFFER_LEN = 10
BUFFER = np.array([-1]*MAX_BUFFER_LEN)
LAST_ACTION = None


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
                actions.sit()
            elif action == 2: # standing
                pass
            else:
                print(f'ERROR! Invalid action {action}')
                return
            
            LAST_ACTION = action

            print(f'{labels[max_i]} has majority with {counts[max_i]*100:.3f}%')
            print(BUFFER)
    # check buffer  
    # if full; continue
    # else; return
    pass


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
    print(action, output_data)
    return action, output_data


def main(tflite_path):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.resize_tensor_input(0, (1, 224, 224, 3), strict=0)
    interpreter.allocate_tensors()

    print('INFO: Finished initializing')

    i = 0
    while(True):
        X = np.random.rand(1, 224, 224, 3) * 255. # capture.get_image()
        # X = tf.keras.applications.mobilenet.preprocess_input(X)
        Y, probs  = make_inference(interpreter, X)
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
