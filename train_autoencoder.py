import tensorflow as tf
import numpy as np
import datetime
import pdb
import autoencoder

# Load MNIST data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/')

x_placeholder = tf.placeholder(tf.float32, [None,28,28,1], name='x_placeholder')

encoded = autoencoder.encoder(x_placeholder)
decoded = autoencoder.decoder(encoded)

loss = tf.reduce_mean(tf.squared_difference(decoded, x_placeholder))

tvars = tf.trainable_variables()
auto_vars = [var for var in tvars if 'a_' in var.name or 'u_' in var.name]

a_trainer = tf.train.AdamOptimizer(0.001).minimize(loss, var_list=auto_vars)

tf.get_variable_scope().reuse_variables()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

num_iterations, batch_size = 100000, 50
for i in range(num_iterations):
    image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
    _, aLoss = sess.run([a_trainer, loss], {x_placeholder: image_batch})
    if (i % 10 == 0):
        print(i)
        print(aLoss)

model_name = 'autoencoder_model1'
save_path = saver.save(sess, 'trained_models/' + model_name + '.ckpt')
print(model_name + ' saved in path: %s' % save_path)



