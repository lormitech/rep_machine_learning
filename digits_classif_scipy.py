
#
# last modified: 23.12.19
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

from sklearn import metrics
from sklearn import svm

from sklearn import datasets


#--

# loading the data (8x8 images of digits)
digits = datasets.load_digits()
print(dir(digits))
#
# note that ...
# to load image files (PNG format), it is possible to use the
# function 'matplotlib.pyplot.imread()'
# https://matplotlib.org/api/_as_gen/matplotlib.pyplot.imread.html
# 

# number of images
no_images = len(digits.images)
str01 = 'no. of images: {0}\n'.format(no_images)
print(str01)

#
# note that ...
# a 'label' is assigned to each image; the label gives the
# digit that the given image represents;
# the labels are stored into the data structure 'digits.target'
#

# combining images and labels ...
images_and_labels = list(zip(digits.images, digits.target))

# to apply a classifier on the data, we need to flatten the images, to
# turn the data in a (samples, features) matrix
no_samples = no_images
data_img = digits.images.reshape((no_samples, -1))

#
# note that ...
# 'to flatten' an image means to convert a 2D array of pixels
# into an 1D array of features (feature vector)
#


# training subset
train_img = data_img[:no_samples // 2]
train_lbl = digits.target[:no_samples // 2]

# test subset
test_img = data_img[no_samples // 2:]
test_lbl = digits.target[no_samples // 2:]

#--

# initializing a classifier (support vector classifier)
classifier = svm.SVC(gamma=0.001)

# learning the digits on the first half of the digits
classifier.fit(train_img, train_lbl)

# expected values of the digit on the second half of the digits
data_expected = test_lbl

# predicting the value of the digit on the second half of the digits
data_predicted = classifier.predict(test_img)

# combining images and predicted values ...
images_and_predictions = list(zip(digits.images[no_samples // 2:], data_predicted))


class_report = metrics.classification_report(data_expected, data_predicted)
str02 = 'classification report for classifier {0}:\n{1}\n'.format(classifier, class_report)
print(str02)

confusion_mx = metrics.confusion_matrix(data_expected, data_predicted)
str03 = 'confusion matrix:\n{0}\n'.format(confusion_mx)
print(str03)

accuracy_score = metrics.accuracy_score(data_expected, data_predicted)
str04 = 'accuracy score:\n{0}\n'.format(accuracy_score)
print(str04)


#--

# let's consider the first 4 images within the two groups ...
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
