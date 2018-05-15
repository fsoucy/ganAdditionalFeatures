"""
This is a straightforward Python implementation of a generative adversarial network.
The code is drawn directly from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

A version of this model with explanatory notes is also available on GitHub
at https://github.com/jonbruner/generative-adversarial-networks.

This script requires TensorFlow and its dependencies in order to run. Please see
the readme for guidance on installing TensorFlow.

This script won't print summary statistics in the terminal during training;
track progress and see sample images in TensorBoard.
"""

import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Load MNIST data
data = np.load('unweighted.npy')

# Define the discriminator network
def discriminator(data, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [10, 10], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [10], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(data, [-1, 10])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [10, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # Third fully connected layer
        d_w5 = tf.get_variable('d_w5', [10, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b5', [1], initializer=tf.constant_initializer(0))
        d5 = tf.matmul(d4, d_w5) + d_b5

        # d4 contains unscaled values
        return d5

# Define the generator network
def generator(z, batch_size, z_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
    g1 = tf.nn.relu(g1)

    g_w2 = tf.get_variable('g_w2', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.matmul(g1, g_w2) + g_b2
    g2 = tf.reshape(g2, [-1, 56, 56, 1])
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='g_b1')
    g2 = tf.nn.relu(g2)

    g_w3 = tf.get_variable('g_w3', [z_dim, 3136], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [3136], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.matmul(g2, g_w3) + g_b3
    g3 = tf.reshape(g3, [-1, 56, 56, 1])
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='g_b1')
    g3 = tf.nn.relu(g3)

    g4 = tf.sigmoid(g3)

    return g4

tf.reset_default_graph()

z_dimensions = 100
batch_size = 50
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,28,28,1], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, batch_size, z_dimensions)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

# Define losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

# Define variable lists
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Define the optimizers
# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

saver = tf.train.Saver()


with tf.Session() as sess:
    # Restore variables from disk.
    saver.restore(sess, "trained_models/model1.ckpt")
    print("Model restored.")
    batch_size = 1000
    z_dimensions = 100
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

    generated_data_output = generator(z_placeholder, batch_size, z_dimensions)
    z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])

    generated_data = sess.run(generated_data_output, feed_dict={z_placeholder: z_batch})
    generated_data = generated_data.reshape([batch_size, 28, 28])
    for i in range(batch_size):
        data_loc = 'generated_data/genDat' + str(i) + '.png'
        generated_datapoint = generated_data[i, :, :]
        plt.imsave(img_loc, generated_datapoint, cmap='Greys')
