'''
Date: 2021-05-14 16:54:03
Author: ZHR
Description: please input the description
LastEditTime: 2021-05-14 20:45:11
'''
import tensorflow as tf
hello = tf.constant('hello,tf')
sess = tf.compat.v1.Session()
print(sess.run(hello))