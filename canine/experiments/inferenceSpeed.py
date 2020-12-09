import TensorRT_test as trtt
import time

#if __name__ == "__main__":

import numpy as np
import tensorflow as tf

models_file = open('./modelNames.txt','r')
modelNames = models_file.readlines()

model_times = open('./modelInferenceResults','w')

for model in modelNames:
    model = model.rstrip("\n")

    interpreter = tf.lite.Interpreter(model_path="/home/nano/Development/canine/models/%s" % (model))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    #print("\n\n\n*************************************************")
    #print("*********** STARTING INFERENCE ENGINE ***********")
    #print("*************************************************")

    #print("warming up")
    #for i in range(50):
    #    interpreter.invoke()
    #print("finished warming up")

    start_time = time.time()

    for i in range(100):
        interpreter.invoke()

    end_time = time.time()
    model_times.write('%s Inference speed: %.2f samples/s\n'%(model, 100*1/(end_time-start_time)))
    print('Inference speed: %.2f samples/s'%(100*1/(end_time-start_time)))
