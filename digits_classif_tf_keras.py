
#
# last modified: 24.07.20
#

#--

#
# classification of hand-written digits
#

#--

import matplotlib as mpl
# configuring the backend for 'matplotlib'
# note that the function 'use()' must be called
# before importing 'matplotlib.pyplot'
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
# TensorFlow vers. 1.x
import tensorflow as tf
from tensorflow import keras


#--

# setting verbosity for the logging system
tf.logging.set_verbosity(tf.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#--

def displayGrayscaleImg(img):
    '''
    function to display a grayscale image

    '''
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.grid(False)
    plt.show()


#--

# loading the data (8x8 images of digits)
digits = datasets.load_digits()
print(dir(digits))

# number of images
no_images = len(digits.images)
str01 = '\nno. of images: {0}'.format(no_images)
print(str01)

# image resolution
res_img = digits.images.shape[1:3]
str02 = '\nimage resolution: {0}'.format(res_img)
print(str02)

#
# note that ...
# a 'label' is assigned to each image; the label gives the
# digit that the given image represents;
# the labels are stored into the data structure 'digits.target'
#

# labels
labels_l = list(dict.fromkeys(digits.target))
str03 = '\nlabels:\n{0}'.format(labels_l)
print(str03)
# number of assigned labels
no_labels = len(digits.target)
str04 = '\nno. of assigned labels: {0}'.format(no_labels)
print(str04)

#--

# displaying an image ...
#displayGrayscaleImg(digits.images[4])


#--

# training subset
train_img = digits.images[:no_images // 2]
train_lbl = digits.target[:no_images // 2]

# test subset
test_img = digits.images[no_images // 2:]
test_lbl = digits.target[no_images // 2:]


#--

# Neural Network (NN) - configuring the layers of the model

#
# let us consider a NN with three layers:
#   1) input layer
#   2) hidden layer
#   3) output layer
# where layers #2 and #3 are 'dense layers'
#
# layer #1 (input layer) returns the image after a flattening procedure;
#
# layer #3 (output layer) returns an array of 10 probability scores that sum to 1;
# each node of the layer contains a score that indicates the probability that the current
# image belongs to one of the 10 classes associated with the labels;
#
#
#
# let us consider a 'dense layer' ...
# a dense layer consists of N neurons;
# the dimension of the output space is given by the number of neurons (N);
# the number of neurons (N) is a 'hyperparameter';
#
# in order to design a dense layer, the number of neurons (N) needs to be set;
#
# for layer #3 (output layer) ...
#   N = n_l3; n_l3 = number of labels; n_l3 = 10
#
# for layer #2 (hidden layer) ...
#   N = n_l2; number of labels < n_l2 < number of pixels; 10 < n_l2 < 64
#
#

# dense layers, dimensionality of output space
n_l2 = 32
n_l3 = no_labels

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(res_img[0], res_img[1])),
    keras.layers.Dense(units=n_l2, activation='relu'),
    keras.layers.Dense(units=n_l3, activation='softmax')
])


# compiling the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#--

#
# considering the digits on the first half of the digits ...
#

# training the model, learning the digits on the first half of the digits
model.fit(train_img, train_lbl, epochs=10)

# evaluating model performance
test_loss, test_acc = model.evaluate(test_img,  test_lbl, verbose=2)
str05 = 'accuracy: {0}\n'.format(test_acc)
print(str05)



#
# considering the digits on the second half of the digits ...
#

# expected values of the digit on the second half of the digits
data_expected = test_lbl

# combining images and labels (expected values) ...
images_and_labels = list(zip(digits.images[no_images // 2:], data_expected))

# predicting the value of the digit on the second half of the digits
predictions = model.predict(test_img)

def getDigitValue_hc(conf_values):
    '''
    function to get the value of the digit with
    the highest confidence value

    '''
    # index associated with the highest
    # confidence value
    idx = np.argmax(conf_values)
    # labels
    labels = np.arange(10)
    dvalue_hc = labels[idx]

    return dvalue_hc

# predicted values of the digit on the second half of the digits
data_predicted = np.apply_along_axis(getDigitValue_hc, 1, predictions)

# combining images and predicted values ...
images_and_predictions = list(zip(digits.images[no_images // 2:], data_predicted))


#--

# let's consider the first 4 images from the test subset and
# compare the expected and predicted values ...
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('training: %i' % label)

for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('prediction: %i' % prediction)

plt.show()
