
#
# last modified: 03.01.21
#

#--

import os

import tensorflow as tf


#--

# setting verbosity for the logging system
#tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel('ERROR')

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#--

# TensorFlow version
tf_vers = tf.__version__
str01 = '\nTensorFlow version: {0}'.format(tf_vers)
print(str01)


# TensorFlow logging system, verbosity level
tf_log_vl = tf.get_logger()
str02 = '\nTensorFlow logging system, verbosity level: {0}'.format(tf_log_vl)
print(str02)



#--

gpu_dev = tf.test.gpu_device_name()
if gpu_dev != '/device:GPU:0':
  str03 = '\nGPU device not found'
  print(str03)
else:
  str04 = '\nGPU device: {0}'.format(gpu_dev)
  print(str04)

