import tensorflow as tf
import numpy as np
import argparse
import time
import os


def test_speed(tflite_path, batch_size, num_batches):
    X = tf.convert_to_tensor(np.random.rand(batch_size, 224, 224, 3), dtype_hint=tf.float32)

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.resize_tensor_input(0, X.shape, strict=0)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        X = (X / input_scale + input_zero_point).astype(input_details['dtype'])

    interpreter.set_tensor(input_details['index'], X)

    start_t = time.time()
    for _ in range(num_batches):
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details['index'])
    end_t = time.time()
    
    total_t = end_t - start_t
    per_sample_t = batch_size*num_batches/total_t
    
    print(f'BATCH_SIZE: {batch_size}, NUM_BATCHES: {num_batches} in {total_t:.3f} seconds or {per_sample_t:.3f}.')
    return total_t, per_sample_t


def main(args):
    if not os.path.isfile(args.path):
        exit(1)
    test_speed(tflite_path=args.path, batch_size=args.batch_size, num_batches=args.num_batches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='data preprocessing')
    parser.add_argument('-p', '--path', metavar='path', type=str,
                        required=True, help='the path to tflite model')
    parser.add_argument('-b', '--batch_size', metavar='batch_size', type=int, default=1,
                        required=False, help='size for a batch of images')
    parser.add_argument('-n', '--num_batches', metavar='num_batches', type=int, default=64,
                        required=False, help='how many times to make an inference')
    args = parser.parse_args()
    main(args)
