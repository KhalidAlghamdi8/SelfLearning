from re import I
from pyrsistent import s
from regex import P
from sklearn import datasets, linear_model
import numpy as np
from sklearn.model_selection import train_test_split

from return_data import read_goog_msft

xData, yData = read_goog_msft()

# Set up a LinearRegression
googModel = linear_model.LinearRegression()
googModel.fit(xData.reshape(-1, 1), yData.reshape(-1, 1))

# Find the coefficients of the linear regression
print(googModel.coef_)
print(googModel.intercept_)

#################################
#
# Simple linear regression algorithm - one point per epochs
#
#################################

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# Model linear regression y = Wx + b
W = tf.Variable(tf.zeros([1,1]))
b = tf.Variable(tf.zeros([1]))

# Placeholder to feed the returns, returns have many rows
# Just one column
x= tf.compat.v1.placeholder(tf.float32, [None,1])
Wx = tf.matmul(x,W)

y = Wx + b

# placeholder to hold the y_labels
y_ = tf.compat.v1.placeholder(tf.float32, [None,1])

cost = tf.reduce_mean(tf.square(y_ - y))

train_test_constant = tf.compat.v1.train.GradientDescentOptimizer(0.1).minimize(cost) 

def train_test_WithOnePoint(steps, train_step):
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

    for i in range(steps):
        xs = np.array([[xData[i % len(xData)]]])
        ys = np.array([[yData[i % len(yData)]]])

        feed = {x: xs, y_: ys}
        sess.run(train_step, feed_dict=feed)
        
        if (i + 1) % 1000 == 0:
            print("After %d iteration:" % i)

            print("W: %f" % sess.run(W))
            print("b: %f" % sess.run(b))

            print("cost: %f" % sess.run(cost, feed_dict=feed))

train_test_WithOnePoint(10000, train_test_constant)
