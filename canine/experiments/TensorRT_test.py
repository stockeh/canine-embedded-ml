import tensorflow as tf
import sys
import numpy as np
import time

from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.saved_model import tag_constants

INPUT_MODEL_DIR = sys.argv[1]

def benchmark_saved_model(SAVED_MODEL_DIR, BATCH_SIZE=64, NUM_BATCHES=100):
    saved_model_loaded = tf.saved_model.load(SAVED_MODEL_DIR, tags=[tag_constants.SERVING])
    signature_keys = list(saved_model_loaded.signatures.keys())
    print(signature_keys)

    infer = saved_model_loaded.signatures['serving_default']
    print(infer.structured_outputs)

    x = tf.convert_to_tensor(np.random.rand(BATCH_SIZE, 224, 224, 3), dtype_hint=tf.float32)
    
    print("\n\n\n*************************************************")
    print("*********** STARTING INFERENCE ENGINE ***********")
    print("*************************************************")
    
    print("warming up")
    for i in range(50):
        labeling = infer(x)
    print("finished warming up")

    start_time = time.time()

    for i in range(NUM_BATCHES):
        labeling = infer(x)
        
    end_time = time.time()
    print('Inference speed: %.2f samples/s'%(NUM_BATCHES*BATCH_SIZE/(end_time-start_time)))

if(len(sys.argv)>2):
    PRECISION = sys.argv[2]
    params = tf.experimental.tensorrt.ConversionParams(
        precision_mode=PRECISION)

    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=INPUT_MODEL_DIR, conversion_params=params)
    converter.convert()
    converter.save(INPUT_MODEL_DIR+'-trt-'+PRECISION)
    INPUT_MODEL_DIR = INPUT_MODEL_DIR+'-trt-'+PRECISION

benchmark_saved_model(INPUT_MODEL_DIR, BATCH_SIZE=1, NUM_BATCHES=64)
