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
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

# Weights and Biases per layer
init_std = 0.2 #Adjust this to randomize the intial values of W matrices
init_bias = 10 #Adjust this to make the bias vectors in small positive values

# Scanning size and number of stacks for each CNV layer
CL1 = 6
CL2 = 12
CL3 = 24
FL = 200 # Fully-connected layer

Wcnv1 = tf.Variable(tf.truncated_normal([5, 5, 1, CL1], stddev=init_std)) 
Wcnv2 = tf.Variable(tf.truncated_normal([4, 4, CL1, CL2], stddev=init_std)) 
Wcnv3 = tf.Variable(tf.truncated_normal([4, 4, CL2, CL3], stddev=init_std))
Wfl = tf.Variable(tf.truncated_normal([7*7*CL3, FL], stddev=init_std))
W = tf.Variable(tf.truncated_normal([FL, 10], stddev=init_std))

#Non-zero initial bias to be applied to RELU
Bcnv1 = tf.Variable(tf.ones([CL1])/init_bias)
Bcnv2 = tf.Variable(tf.ones([CL2])/init_bias)
Bcnv3 = tf.Variable(tf.ones([CL3])/init_bias)
Bfl = tf.Variable(tf.ones([FL])/init_bias) 
B = tf.Variable(tf.ones([10])/init_bias)


# CNV layers
stride = 1 #Output is 28x28
Ycnv1 = tf.nn.conv2d(X, Wcnv1, strides=[1, stride, stride, 1], padding='SAME') #Something like matrix-multiplication
Y2 = tf.nn.relu(Ycnv1 + Bcnv1)
stride = 2 #Output is 14x14
Ycnv2 = tf.nn.conv2d(Y2, Wcnv2, strides=[1, stride, stride, 1], padding='SAME')
Y3 =tf.nn.relu(Ycnv2 + Bcnv2)
stride = 2 #Output is 7x7
Ycnv3 = tf.nn.conv2d(Y3, Wcnv3, strides=[1, stride, stride, 1], padding='SAME')
Y4 = tf.nn.relu(Ycnv3 + Bcnv3)

#Fully-connected layer
pkeep = tf.placeholder(tf.float32)
YY4 = tf.reshape(Y4, [-1, 7*7*CL3])
Y5 = tf.nn.relu(tf.matmul(YY4, Wfl) + Bfl)
Y5d = tf.nn.dropout(Y5, pkeep)

#Output layer
Ylogits = tf.matmul(Y5d, W) + B
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
allweights = tf.concat(0, [tf.reshape(Wcnv1, [-1]), tf.reshape(Wcnv2, [-1]), tf.reshape(Wcnv3, [-1]), tf.reshape(Wfl, [-1]), tf.reshape(W, [-1])])
allbiases  = tf.concat(0, [tf.reshape(Bcnv1, [-1]), tf.reshape(Bcnv2, [-1]), tf.reshape(Bcnv3, [-1]), tf.reshape(Bfl, [-1]), tf.reshape(B, [-1])])
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
        a, c, im, w, b = sess.run([accuracy, cross_entropy, I, allweights, allbiases], feed_dict={X: batch_X, Y_: batch_Y, pkeep: 1.0})
        datavis.append_training_curves_data(i, a, c)
        datavis.append_data_histograms(i, w, b)
        datavis.update_image1(im)
        print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c)) # Test loss = Cost-function = Cross-entropy

    # Report statistics for test set at iteration i
    if update_test_data:
        a, c, im = sess.run([accuracy, cross_entropy, It], feed_dict={X: mnist.test.images, Y_: mnist.test.labels, pkeep: 1.0})
        datavis.append_test_curves_data(i, a, c)
        datavis.update_image2(im)
        print(str(i) + ": ********* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ********* test accuracy:" + str(a) + " test loss: " + str(c))

    #Learning Rate Decay. Start fast (exp(-1/2000) = 0.99) and decay the learning rate exponentially
    lr_max = 0.003
    lr_min = 0.0001
    learning_rate = lr_min + (lr_max - lr_min) * np.exp(-i/2000)

    # Gradient Descent
    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y, lr: learning_rate, pkeep: 0.7})

n_iters = 3000
datavis.animate(training_step, iterations=n_iters+1, train_data_update_freq=10, test_data_update_freq=50, more_tests_at_start=True)

print("Max test accuracy: " + str(datavis.get_max_test_accuracy()))

