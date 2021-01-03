
#
# last modified: 03.01.21
#

#--

import os

import tensorflow as tf
import tensorflow_probability as tfp


#--

# setting verbosity for the logging system
#tf.get_logger().setLevel('INFO')
tf.get_logger().setLevel('ERROR')

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#--

# generating data ...
features = tfp.distributions.Normal(loc=0., scale=1.).sample(int(100e3))
labels = tfp.distributions.Bernoulli(logits=1.618 * features).sample()

# defining a model, a Generalized Linear Model (GLM)
model = tfp.glm.Bernoulli()

# fitting the model to the given data
coeffs, linear_response, is_converged, no_iter = tfp.glm.fit(
    model_matrix=features[:, tf.newaxis],
    response=tf.cast(labels, dtype=tf.float32),
    model=model)


str01 = '\nmodel coefficients: {0}'.format(coeffs)
print(str01)
str02 = '\npredicted linear response: {0}'.format(linear_response)
print(str02)
str03 = '\nconvergence criteria: {0}'.format(is_converged)
print(str03)
str04 = '\nnumber of iterations: {0}'.format(no_iter)
print(str04)
