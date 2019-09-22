
#
# last modified: 09.08.19
#

#--

import numpy as np
import tensorflow as tf


#--

# setting verbosity for the logging system
tf.logging.set_verbosity(tf.logging.ERROR)


#--

# iterations

no_iter = 5
x1 = tf.Variable(0, name='x1')
model = tf.global_variables_initializer()

with tf.Session() as session:
    for i in range(no_iter):
        session.run(model)
        x1 = x1 + 1
        x1_e = session.run(x1)
        print(x1_e)



#--

# iterations, constrains

x2 = tf.Variable(0., name='x2')
threshold = tf.constant(7.)

model = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(model)
    while session.run(tf.less(x2, threshold)):
        x2 = x2 + 1
        x2_e = session.run(x2)
        print(x2_e)



#--

# data wrangling: shaping

no_dpoints = 8
ds01 = np.arange(no_dpoints)
print(ds01)

# defining a 3D data structure (rank-3 tensor)
ds02 = tf.reshape(ds01, [2, 2, 2])

# unpacking the rank-3 tensor into two rank-2 tensors
# along the dimension #1
ds03 = tf.unstack(ds02, axis=1)

# combining the two rank-2 tensors into a rank-3 tensor
# along the dimension #1
ds04 = tf.stack(ds03, axis=1)

with tf.Session() as session:
        ds02_e = session.run(ds02)
        print('\nds02:')
        print(ds02_e)
        print(ds02_e[0])
        ds03_e = session.run(ds03)
        print('\nds03:')
        print(ds03_e)
        print(ds03_e[0])
        ds04_e = session.run(ds04)
        print('\nds04:')
        print(ds04_e)
        print(ds04_e[0])
