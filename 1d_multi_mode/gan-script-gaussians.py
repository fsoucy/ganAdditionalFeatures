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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load MNIST data
#data = np.load('3_1.npy')
#data = data.reshape((data.shape[0], 1))
# real_data = np.random.normal(loc=-0.6, scale=0.7, size=10000).reshape((10000, 1))
real_data = np.load('data.npy').reshape((12000,1))

# Define the discriminator network
def discriminator(data, reuse_variables=None):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:

        # First fully connected layer
        d_w3 = tf.get_variable('d_w3', [1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [32], initializer=tf.constant_initializer(0))
        _data = tf.reshape(data, [-1, 1])
        d3 = tf.matmul(_data, d_w3) + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [32, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4

# Define the generator network
def generator(z, batch_size, z_dim):
    g_w1 = tf.get_variable('g_w1', [z_dim, 32], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    # g1 = tf.reshape(g1, [-1, 1])
    g1 = tf.nn.relu(g1)

    g_w2 = tf.get_variable('g_w2', [32, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.matmul(g1, g_w2) + g_b2

    return g2

tf.reset_default_graph()

z_dimensions = 1
batch_size = 25
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,1], name='x_placeholder')
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
d_trainer_fake = tf.train.AdamOptimizer(0.008).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.008).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.004).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())


# Pre-train discriminator
pre_train_iterations = 1000
for i in range(pre_train_iterations):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    np.random.shuffle(real_data)
    real_batch = real_data[0:batch_size, :]
    generated = sess.run([Gz], {z_placeholder: z_batch})[0]
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake], {x_placeholder: real_batch, z_placeholder: z_batch})


iterations = 5000
for i in range(iterations):
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    np.random.shuffle(real_data)
    real_batch = real_data[0:batch_size, :]
    generated = sess.run([Gz], {z_placeholder: z_batch})[0]
    if (i % 100 == 0):
        print(i)
        print(generated.mean())
        print(generated.std())
        print(real_batch.mean())
        print(real_batch.std())

    # Train discriminator
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake], {x_placeholder: real_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, z_dimensions])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})


model_name = 'modelGaussiansSingle'

save_path = saver.save(sess, 'trained_models/' + model_name + '.ckpt')
print(model_name + " saved in path: %s" % save_path)


with tf.Session() as sess:
    saver.restore(sess, 'trained_models/modelGaussiansSingle.ckpt')
    print("Model restored.")
    batch_size = 1000
    z_dimensions = 1
    z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions])

    gen = generator(z_placeholder, batch_size, z_dimensions)
    z_batch = np.random.normal(0, 1, [batch_size, z_dimensions])

    genOutput = sess.run(gen, feed_dict={z_placeholder: z_batch})
    genOutput = genOutput.reshape([batch_size, 1])
    print(genOutput.mean())
    np.save('genLowVariance.npy', genOutput)

print(genOutput.mean())
print(genOutput.std())
smallData = real_data[0:1000, :]
bins = np.linspace(1, 32, 100)
plt.hist(genOutput, bins, alpha=0.5, label='Generated')
plt.hist(smallData, bins, alpha=0.5, label='Data')
plt.legend(loc='upper right')
plt.savefig('test3.png')
