#!/usr/bin/env python
# coding=utf-8

'''
@author: 范禄林
@contact: fxjzzyo@163.com
@time: 2018/3/28 19:42
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
x = np.linspace(-1,1,100)[:,np.newaxis] # (100,1)
noise = np.random.normal(0,0.1,x.shape)
y = np.power(x,2)+noise

# input placeholder
tf_x = tf.placeholder(tf.float32,x.shape,name='x-input')
tf_y = tf.placeholder(tf.float32,y.shape,name='y-output')

# net
l1 = tf.layers.dense(tf_x,10,activation=tf.nn.relu)# the hidden layer
out = tf.layers.dense(l1,1) # output layer

# loss
loss = tf.losses.mean_squared_error(tf_y,out)
optimizer = tf.train.GradientDescentOptimizer(0.5)# optimizer
train = optimizer.minimize(loss)# train

#session
sess = tf.Session()

# initialize variable
sess.run(tf.global_variables_initializer())

# train
for step in range(101):
    _,loss_,pre = sess.run([train,loss,out],feed_dict={tf_x:x,tf_y:y})
    if step % 10 == 0:
        plt.cla()
        plt.scatter(x,y)
        plt.plot(x,pre,'r-',lw=4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('the regression test')
        plt.text(-0.2,0.8,'after %d train, loss is %.4f' %(step,loss_))
        plt.pause(0.2)
        print('after %d train, loss is %.4f' %(step,loss_))
sess.close()
plt.show()