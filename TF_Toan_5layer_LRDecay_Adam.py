# encoding: UTF-8
# Copyright 2016 Google.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exercise attempt by Toan Luong 2/3/2017

import tensorflow as tf
import numpy as np
import tensorflowvisu
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
tf.set_random_seed(0)

# Download data
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# Create a placeholder to hold different tensors
X = tf.placeholder(tf.float32, [None, 28, 28, 1]) # Original X-tensor
Y_ = tf.placeholder(tf.float32, [None, 10]) # Correct labels

# Number of neurons per layer
L2 = 200
L3 = 100
L4 = 60
L5 = 30
L6 = 10 #Final 

# Weights and Biases per layer
init_std = 0.2 #Adjust this to randomize the intial values of W matrices
init_bias = 10 #Adjust this to make the bias vectors in small positive values, instead of 0, so that neurons operate in the non-zero range of RELU

W1 = tf.Variable(tf.truncated_normal([28*28, L2], stddev=init_std)) # tf.truncated_normal is a TensorFlow function that produces random values following the normal (Gaussian) distribution between -2*stddev and +2*stddev.
W2 = tf.Variable(tf.truncated_normal([L2, L3], stddev=init_std)) # Initialize random values instead of 0
W3 = tf.Variable(tf.truncated_normal([L3, L4], stddev=init_std))
W4 = tf.Variable(tf.truncated_normal([L4, L5], stddev=init_std))
W5 = tf.Variable(tf.truncated_normal([L5, L6], stddev=init_std))

#Non-zero initial bias to be applied to RELU
B1 = tf.Variable(tf.ones([L2])/init_bias)
B2 = tf.Variable(tf.ones([L3])/init_bias)
B3 = tf.Variable(tf.ones([L4])/init_bias)
B4 = tf.Variable(tf.ones([L5])/init_bias) 
B5 = tf.Variable(tf.ones([L6])/init_bias)


# Model. Relu as activation function. The sigmoid activation function is actually quite problematic in deep networks. 
# It squashes all values between 0 and 1 and when you do so repeatedly, neuron outputs and their gradients can vanish entirely. 

XX = tf.reshape(X, [-1, 784]) # Flatten the images into a single line of pixels. '-1' = preserve that dimension
Y2 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y3 = tf.nn.relu(tf.matmul(Y2, W2) + B2)
Y4 = tf.nn.relu(tf.matmul(Y3, W3) + B3)
Y5 = tf.nn.relu(tf.matmul(Y4, W4) + B4)
Ylogits = tf.matmul(Y5, W5) + B5 #This is to be passed into cross_entropy to avoid log(0) error
Ypred = tf.nn.softmax(Ylogits)

#Define cross-entropy (the mean distance between Ypred and Y_)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

#Train using Adam Optimizer. 
lr = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# Training statistics
correct_prediction = tf.equal(tf.argmax(Ypred, 1), tf.argmax(Y_, 1)) # Compare the two predicted/actual columns, element-wise. tf.argmax(arr, 0) to return the index of the maximum element per row.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # Computes the mean of elements across dimensions of the tensor (Correct_prediction)

# Visualization
allweights = tf.concat(0, [tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]), tf.reshape(W5, [-1])])
allbiases  = tf.concat(0, [tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]), tf.reshape(B5, [-1])])
I = tensorflowvisu.tf_format_mnist_images(X, Ypred, Y_)  # assembles 10x10 images by default
It = tensorflowvisu.tf_format_mnist_images(X, Ypred, Y_, 1000, lines=25)  # 1000 images on 25 lines
datavis = tensorflowvisu.MnistDataVis()

# Run
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# Training function to be put in a loop at the end. i is loop-count
def training_step(i, update_test_data, update_train_data):

    # Split into mini-batches
    batch_size = 100
    batch_X, batch_Y = mnist.train.next_batch(batch_size)

    # Report statistics for training set at iteration i
    if update_train_data:
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c)) # Test loss = Cost-function = Cross-entropy

    # Report statistics for test set at iteration i
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    #Learning Rate Decay. Start fast (exp(-1/2000) = 0.99) and decay the learning rate exponentially
    lr_max = 0.003
    lr_min = 0.0001
    learning_rate = lr_min + (lr_max - lr_min) * np.exp(-i/2000)

    # Gradient Descent
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate})

n_iters = 2000
datavis.animate(training_step, iterations=n_iters+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

print("Max test accuracy: " + str(datavis.get_max_test_accuracy()))

