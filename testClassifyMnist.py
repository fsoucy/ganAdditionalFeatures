from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import mnist_classifier
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import scipy

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
FLAGS, unparsed = parser.parse_known_args()

mnist = input_data.read_data_sets(FLAGS.data_dir)

batch_size = mnist.train.images[0:1000]
initial_dist = mnist_classifier.summary_statistics(batch_size)
print(initial_dist)

batch_size = batch_size.reshape([1000, 28, 28])

for i in range(1000):
    img_loc = 'debug_mnist_images/img' + str(i) + '.npy'
    img = batch_size[i, :, :]
    np.save(img_loc, img)


x, images = glob.glob('debug_mnist_images/*'), []

for loc in x:
    y = np.load(loc)
    loc = 'debug_processed_mnist_images/' + loc.split('/')[1].split('.')[0] + '.npy'
    np.save(loc, y)
    y = np.reshape(y, [784])
    images.append(y)

images = np.array(images)
new_stats = mnist_classifier.summary_statistics(images)
print(new_stats)
