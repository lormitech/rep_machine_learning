
#
# last modified: 04.02.20
#

#--

#
# regression
#


#--

import numpy as np
import tensorflow as tf


#--

# setting verbosity for the logging system
tf.logging.set_verbosity(tf.logging.ERROR)


#--

def makeLinearData(x, a, b):
    return a + b * x


#--

# generating training data ...
no_dpoints = 50
xdata = np.linspace(0, 4, no_dpoints)
y = makeLinearData(xdata, 2, 1)
np.random.seed(1729)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise

# training data (50 data points)
train_data = [xdata, ydata]
print(train_data)

# defining a linear model
w = tf.Variable([1.5], dtype=tf.float32)
b = tf.Variable([2.5], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w * x + b

# defining the loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# learning rate
learning_rate = 0.001
# initializing the optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
# defining an operation to minimize the loss function
op_min = optimizer.minimize(loss)

# starting a session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # running the training 1000 times
    for _ in range(1000):
        sess.run(op_min, {x: train_data[0], y: train_data[1]})

    # getting optimized values for 'w' and 'b' with the trained model
    result_e = sess.run([w, b])
    w_optimal = result_e[0][0]
    b_optimal = result_e[1][0]
    print(w_optimal, b_optimal)

    #--

    # evaluating the predicted 51th 'y' value corrisponding to x = (4.0 + 0.08163266)
    y_pred_e = sess.run(linear_model, {x: [4.081633]})
    print(y_pred_e)
