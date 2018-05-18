# Autoencoder for MNIST images.

import tensorflow as tf
import numpy as np
import datetime
import pdb

# Load MNIST data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

def encoder(images, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # Input should be (-1, 28, 28, 1)
        # First convolutional layer
        a_w1 = tf.get_variable('a_w1', [5, 5, 1, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        a_b1 = tf.get_variable('a_b1', [16], initializer=tf.constant_initializer(0))
        a1 = tf.nn.conv2d(input=images, filter=a_w1, strides=[1,1,1,1], padding='SAME')
        a1 = tf.nn.relu(a1 + a_b1)
        a1 = tf.nn.avg_pool(a1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # Now should be (-1, 14, 14, 16)
        # Second convolutional layer
        a_w2 = tf.get_variable('a_w2', [5,5,16,8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        a_b2 = tf.get_variable('a_b2', [8], initializer=tf.constant_initializer(0))
        a2 = tf.nn.conv2d(input=a1, filter=a_w2, strides=[1,1,1,1], padding='SAME')
        a2 = tf.nn.relu(a2 + a_b2)
        a2 = tf.nn.avg_pool(a2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        
        # Now should be (-1, 7, 7, 8)
        a_w3 = tf.get_variable('a_w3', [3,3,8,4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        a_b3 = tf.get_variable('a_b3', [4], initializer=tf.constant_initializer(0))
        a3 = tf.nn.conv2d(input=a2, filter=a_w3, strides=[1,1,1,1], padding='SAME')
        a3 = tf.nn.relu(a3 + a_b3)

        # Now should be (-1, 7, 7, 4)
        a4 = tf.reshape(a3, [-1, 7 * 7 * 4])
        a_w4 = tf.get_variable('a_w4', [7*7*4, 49], initializer=tf.truncated_normal_initializer(stddev=0.02))
        a_b4 = tf.get_variable('a_b4', [49], initializer=tf.constant_initializer(0))
        a5 = tf.matmul(a4, a_w4) + a_b4 # Now should be (-1, 49)
        return a5

def decoder(encoding, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:
        # Input should be (-1, 49)
        u_w1 = tf.get_variable('u_w1', [49, 7*7*4], initializer=tf.truncated_normal_initializer(stddev=0.02))
        u_b1 = tf.get_variable('u_b1', [7*7*4], initializer=tf.constant_initializer(0))
        u1 = tf.matmul(encoding, u_w1) + u_b1
        u1 = tf.nn.relu(u1)

        # Should be (-1, 7*7*4)
        u1 = tf.reshape(u1, [-1, 7, 7, 4])
        u_w2 = tf.get_variable('u_w2', [3, 3, 4, 8], initializer=tf.truncated_normal_initializer(stddev=0.02))
        u_b2 = tf.get_variable('u_b2', [8], initializer=tf.constant_initializer(0))
        u2 = tf.nn.conv2d(input=u1, filter=u_w2, strides=[1,1,1,1], padding='SAME')
        u2 = tf.nn.relu(u2 + u_b2)

        # Should be (-1, 7, 7, 8)
        u3 = tf.image.resize_images(u2, [14, 14]) # Should now be (-1, 14, 14, 8)
        u_w3 = tf.get_variable('u_w3', [5, 5, 8, 16], initializer=tf.truncated_normal_initializer(stddev=0.02))
        u_b3 = tf.get_variable('u_b3', [16], initializer=tf.constant_initializer(0))
        u3 = tf.nn.conv2d(input=u3, filter=u_w3, strides=[1,1,1,1], padding='SAME')
        u3 = tf.nn.relu(u3 + u_b3)

        # Should be (-1, 14, 14, 16)
        u4 = tf.image.resize_images(u3, [28, 28]) # Should now be (-1, 28, 28, 16)
        u_w4 = tf.get_variable('u_w4', [5, 5, 16, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        u_b4 = tf.get_variable('u_b4', [1], initializer=tf.constant_initializer(0))
        u4 = tf.nn.conv2d(input=u4, filter=u_w4, strides=[1,1,1,1], padding='SAME')
        u4 = tf.nn.relu(u4 + u_b4) # Should now be (-1, 28, 28, 1)

        u5 = tf.reshape(u4, [-1, 28, 28, 1])
        return u5
