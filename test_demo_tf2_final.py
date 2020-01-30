import argparse
import sys
import time
import json
import os
from base64 import b64encode, b64decode

import numpy as np
import tensorflow as tf

from azureml.core.model import Model

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def load_labels(label_file):
    labels = []

    proto_as_ascii_lines = tf.io.gfile.GFile(label_file).readlines()

    for l in proto_as_ascii_lines:
        labels.append(l.rstrip())
    
    return labels

def load_image(image_string):
    base64_bytes = b64decode(image_string)

    image_reader = tf.image.decode_jpeg(base64_bytes, channels=3, name='jpeg_reader')

    float_caster = tf.cast(image_reader, tf.float32)

    dims_expander = tf.expand_dims(float_caster, 0)

    return tf.compat.v1.image.resize_bilinear(dims_expander, [224, 224])

def normalize_image(image_path):
    image_string = tf.io.read_file(image_path)

    image_decoded = tf.image.decode_jpeg(image_string, channels=3)

    image_resized = tf.image.resize(image_decoded, [224, 224])

    return image_resized

def init():
    global graph
    model_path = Model.get_model_path(model_name='pet-detector')
    graph = load_graph(os.path.join(model_path, 'output_graph.pb'))

def run(raw_data=None):
    input_name = 'import/input'
    output_name = 'import/final_result'

    if raw_data is None:
        # running locally
        # fixed model path
        model_path = 'models/output_graph.pb'

        if not os.path.exists(model_path):
            return

        print("Loading graph...")
        graph = load_graph(model_path)

        # fixed JPEG image
        image_path = 'samoyed.jpg'

        if not os.path.exists(image_path):
            return None
        
        print("Normalizing image...")
        resized_image = normalize_image(image_path)
    else:
        # turn raw data (json) into JPEG image
        resized_image = load_image(json.loads(raw_data)['image'])

    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
        

    with tf.compat.v1.Session(graph=graph) as sess:
        start_time = time.time()

        # get numpy array from tensor
        resized_image_array = resized_image.numpy()

        print("Predicting and fetching results...")

        results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: [resized_image_array]})

        end_time = time.time()

    results = np.squeeze(results)

    # return the top results
    sorted_results = results.argsort()[-5:][::-1]
    
    if raw_data is None:
        labels_path = 'models/output_labels.txt'

        if not os.path.exists(labels_path):
            return None
        
        labels = load_labels(labels_path)
    else:
        model_path = Model.get_model_path(model_name='pet-detector')
        labels = load_labels(os.path.join(model_path, 'output_labels.txt'))

    predictions = []

    for i in sorted_results:
        predictions.append(f"{labels[i]} (score={round(results[i], 5)})")

    struct = {
        'evaluation_time': f'Evaluation time (1-image): {end_time - start_time}',
        'predictions': predictions
    }

    return json.dumps(struct)

if __name__ == "__main__":
    print(run())