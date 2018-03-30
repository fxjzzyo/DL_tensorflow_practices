#!/usr/bin/env python
# coding=utf-8

'''
@author: 范禄林
@contact: fxjzzyo@163.com
@time: 2018/3/30 21:00
'''
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#random seed
tf.set_random_seed(1)
np.random.seed(1)

# prepare dataset
mnist = input_data.read_data_sets('../mnist',one_hot=True)
# test data
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# placeholder
with tf.variable_scope('input'):
    tf_x = tf.placeholder(tf.float32,[None,28*28],name='x-input')
    image = tf.reshape(tf_x,[-1,28,28,1])
    tf_y = tf.placeholder(tf.float32,(None,10),name='y-output')

with tf.variable_scope('cnn_net'):
    # cnn layers
    conv1 = tf.layers.conv2d(
        inputs=image,# (28,28,1)
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        name='conv1'
    )# (28,28,16)
    # pooling
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,# (28,28,16)
        pool_size=2,
        strides=2,
        name='pool1'
    )# (14,14,16)

    # conv2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        name='conv2'
    )# （14,14,32）

    # pool2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=2,
        strides=2,
        name='pool2'
    )# (7,7,32)

    # flat
    flat = tf.reshape(pool2,[-1,7*7*32],name='flat')# (7*7*32,1)
    # out
    output = tf.layers.dense(flat,10,name='output')#(10,1)
    # add to histogram summary
    tf.summary.histogram('conv1', conv1)
    tf.summary.histogram('pool1', pool1)
    tf.summary.histogram('conv2', conv2)
    tf.summary.histogram('pool2', pool2)
    tf.summary.histogram('flat', flat)
    tf.summary.histogram('output', output)

# loss
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output,scope='loss')
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
tf.summary.scalar('loss', loss)     # add loss to scalar summary

# accuracy
accuracy = tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1))[1]

# session
sess = tf.Session()
# initial variables
sess.run(tf.global_variables_initializer())# global variable
sess.run(tf.local_variables_initializer())# initial for accuracy

writer = tf.summary.FileWriter('../log', sess.graph)     # write to file
merge_op = tf.summary.merge_all()                       # operation to merge all summary

# train steps
for step in range(601):
    b_x,b_y = mnist.train.next_batch(50)
    _,l,acc,summery = sess.run([train,loss,accuracy,merge_op],feed_dict={tf_x:b_x,tf_y:b_y})
    writer.add_summary(summery,step)
    if step % 50 == 0:
        print('after %d steps, the train losses is %.4f,and the train accuracy is %.4f.' %(step,l,acc))

# test
test_out = sess.run(output,feed_dict={tf_x:test_x[:10]})
pre = np.argmax(test_out,1)
real = np.argmax(test_y[:10],1)
print('the prediction is ',pre)
print('the real number is ',real)