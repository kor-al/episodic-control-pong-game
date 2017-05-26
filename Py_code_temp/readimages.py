
import matplotlib as mpl
import numpy as np
import os
import tensorflow as tf
import time
import seaborn as sns
from PIL import Image
import glob
#
#
from matplotlib import pyplot as plt
from scipy.misc import imsave
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# def next_batch(num, data):
#     """
#     Return a total of `num` samples from the array `data`.
#     """
#     idx = np.arange(0, len(data))  # get all possible indexes
#     np.random.shuffle(idx)  # shuffle indexes
#     idx = idx[0:num]  # use only `num` random indexes
#     data_shuffle = [data[i] for i in idx]  # get list of `num` random samples
#     data_shuffle = np.asarray(data_shuffle)  # get back numpy array
#
#     return data_shuffle
#
#
# # demo data, 1d and 2d array
# Xtr, Ytr = np.arange(0, 10), np.arange(0, 100).reshape(10, 10)
# print(Xtr)
# print(Ytr)
#
# print("\n5 randnom samples from 1d array:")
# print(next_batch(5, Xtr))
# print("5 randnom samples from 2d array:")
# print(next_batch(5, Ytr))

def next_batch(batch_size, image):
    """
    Return a total of `num` samples from the array `data`.
    """
    num_preprocess_threads = 5
    min_queue_examples = 256
    #min_queue_examples = 256

    images = tf.train.shuffle_batch([image],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)

    return images


def read_data(path, num):
     #imglist = [path+("%i.png" % i) for i in range(1,num)]
     imglist = glob.glob(path+"*.png")
     filename_queue = tf.train.string_input_producer(imglist,shuffle=True)
     reader = tf.WholeFileReader()
     key, value = reader.read(filename_queue)
     image = tf.image.decode_png(value) # use png or jpg decoder based on your files.
     image.set_shape((84, 84, 1))
     image = tf.to_float(image, name='ToFloat')
     return image

path = r"D:/Alice/Documents/HSE/masters/observations/"


# imglist = [path+("%i.png" % i) for i in range(1,9)]
# filename_queue = tf.train.string_input_producer(imglist)
#
# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)
#
# my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

image = read_data(path, 10000)
images_batch = next_batch(5, image)
init_op = tf.initialize_all_variables()


# with tf.Session() as sess:
#   sess.run(init_op)
#
#   # Start populating the filename queue.
#   coord = tf.train.Coordinator()
#   threads = tf.train.start_queue_runners(coord=coord)
#   print(image.eval().shape)
#   print(images_batch.eval().shape)
#
#   coord.request_stop()
#   coord.join(threads)

sess = tf.InteractiveSession()

sess.run(init_op)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
image.eval()
for i in range(0,6):
      np_x = images_batch.eval()
      print(np_x.shape)
coord.request_stop()
coord.join(threads)


#images_batch = next_batch(1, image)

#Generate batch
#
# sess = tf.InteractiveSession()
#
# tf.global_variables_initializer()
#
# image = read_data(path, 9)
#
# print(image.eval())

#
#
# print(sub.eval())
#
#
#
# print (batch_xs)
# with tf.Session() as sess:
#     # Required to get the filename matching to run.
#     tf.global_variables_initializer().run()
#
#     # Coordinate the loading of image files.
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#
#     image_tensor = sess.run([image])
#
#     print(image_tensor)
#
#     coord.request_stop()
#     coord.join(threads)