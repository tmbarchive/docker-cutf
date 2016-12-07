# coding: utf-8

from numpy import *
import sys
import os
import os.path
import h5py
import tensorflow as tf
slim = tf.contrib.slim
losses = slim.losses
ops = slim.ops
import tfndlstm as ndlstm
import tfspecs as specs
from urllib2 import urlopen
from collections import namedtuple

# Downoad the MNIST dataset in HDF5 format if necessary.

url = "http://www.tmbdev.net/ocrdata-hdf5/mnist.h5"
if not os.path.exists("mnist.h5"):
    data = urlopen(url).read()
    with open("mnist.h5", "wb") as stream:
        stream.write(data)


# Read the dataset and transform it in standard TensorFlow image batch format (BHWD)

Dataset = namedtuple("Dataset", "images classes size")

def loadh5(fname, prefix=""):
    h5 = h5py.File(fname)
    images = array(h5[prefix+"images"], "f")
    images.shape = images.shape + (1,)
    labels = array(h5[prefix+"labels"], "i")
    del h5
    return Dataset(images, labels, len(images))

train = loadh5("mnist.h5")
test = loadh5("mnist.h5", "test_")

def info(dataset):
    images = dataset.images
    labels = dataset.classes
    print images.shape, amin(images), amax(images)
    print labels.shape, amin(labels), amax(labels)

info(train)
info(test)


# The model definition

with specs.ops:
    model = Cr(32, 3) | Mp(2) | Cr(64, 3) | Mp(2) | Flat | Fr(100) | Fl(10)


# This defines the standard TensorFlow framework for training a network.

sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.float32, [None, 28, 28, 1])
labels = tf.placeholder(tf.int32, [None])

outputs = model.funcall(inputs)

targets = tf.one_hot(labels, 10, 1.0, 0.0)
loss = tf.reduce_sum(tf.square(targets-tf.nn.sigmoid(outputs)))
optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss)

errors = tf.not_equal(tf.argmax(outputs,1), tf.argmax(targets,1))
nerrors = tf.reduce_sum(tf.cast(errors, tf.float32))

sess.run(tf.initialize_all_variables())


def train_epoch(train=train, bs=20):
    total = 0
    count = 0
    for i in range(0, train.size, bs):
        batch_images = train.images[i:i+bs]
        batch_classes = train.classes[i:i+bs]
        feed_dict = {
            inputs: batch_images,
            labels: batch_classes,
        }
        k, _ = sess.run([nerrors, train_op], feed_dict=feed_dict)
        total += k
        count += len(batch_images)
    training_error = total * 1.0 / count
    return count, training_error


def evaluate(test=test):
    bs = 1000
    total = 0
    for i in range(0, test.size, bs):
        batch_images = test[0][i:i+bs]
        batch_classes = test[1][i:i+bs]
        feed_dict = {
            inputs: batch_images,
            labels: batch_classes,
        }
        k, = sess.run([nerrors], feed_dict=feed_dict)
        total += k
    test_error = total * 1.0 / test.size
    return total, test_error


# Now train for 100 epochs and print the training and test set error.

for epoch in range(20):
    _, training_err = train_epoch()
    _, testing_err = evaluate()
    print epoch, training_err, testing_err

