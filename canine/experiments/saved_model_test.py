import tensorflow as tf
import numpy as np
import argparse
import time
import os

from tqdm import tqdm

def test_speed(saved_model_path, batch_size, num_batches):
    X = tf.convert_to_tensor(np.random.rand(batch_size, 224, 224, 3), dtype_hint=tf.float32)

    model = tf.keras.models.load_model(saved_model_path)

    start_t = time.time()
    for _ in tqdm(range(num_batches)):
        output_data = model.predict(X)
    end_t = time.time()

    total_t = end_t - start_t
    per_sample_t = batch_size*num_batches/total_t

    print(f'BATCH_SIZE: {batch_size}, NUM_BATCHES: {num_batches} in {total_t:.3f} seconds or {per_sample_t:.3f}.')
    return total_t, per_sample_t


def main(args):
    test_speed(saved_model_path=args.path, batch_size=args.batch_size, num_batches=args.num_batches)

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
