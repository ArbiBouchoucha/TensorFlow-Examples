'''
Some basic calculations using placeholder,
and then calling it on real data through feed_dict.

Author: Arbi Bouchoucha
Project: https://github.com/ArbiBouchoucha/TensorFlow-Examples
'''

import tensorflow as tf

#setup placeholder using tf.placeholder
x = tf.placeholder(tf.int32, shape=[3], name='x')

''' It is of type integer and it has shape 3 meaning it is a 1D vector with 3 elements in it
we name it x. just create another placeholder y with same dimension. we treat the 
placeholders like we treate constants. '''
y = tf.placeholder(tf.int32, shape=[3], name='y')


sum_x = tf.reduce_sum(x, name="sum_x")
prod_y = tf.reduce_prod(y, name="prod_y")

''' We dont know what values x and y holds till we run the graph '''
final_div = tf.div(sum_x, prod_y, name="final_div")
final_mean = tf.reduce_mean([sum_x, prod_y], name="final_mean")

# Initialize all variables defined
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print("sum(x): ", sess.run(sum_x, feed_dict={x: [100,200,300]}))
    print("prod(y): ", sess.run(prod_y, feed_dict={y: [1,2,3]}))
    print("final_div: ", sess.run(final_div, feed_dict={x: [100,200,300], y: [1,2,3]}))
    print("final_mean: ", sess.run(final_mean, feed_dict={x: [100,200,300], y: [1,2,3]}))

    sess.close()
