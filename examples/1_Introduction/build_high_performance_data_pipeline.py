'''
This python script is implemented following this link:
http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/

Author: Arbi Bouchoucha
Project: https://github.com/ArbiBouchoucha/TensorFlow-Examples
'''

import tensorflow as tf
import numpy as np

x = np.arange(0, 10)

# Create dataset object from numpy array
dx = tf.data.Dataset.from_tensor_slices(x)
dx_batch = tf.data.Dataset.from_tensor_slices(x).batch(3)

# Create a one-shot iterator
''' iterator = dx.make_one_shot_iterator() '''
iterator = dx.make_initializable_iterator()
iterator_batch = dx_batch.make_initializable_iterator()

# Extract an element
next_element = iterator.get_next()
next_element_batch = iterator_batch.get_next()


# Without batch
def simple_example_without_batch():
    print("*** Without any batch: ")
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        ''' Note that the previous statement (i.e., sess.run(iterator.initializer) ) is required to get 
        your iterator ready for action and if you don’t do this before running the 
        next_element operation it will throw an error'''
        for i in range(15):
            val = sess.run(next_element)
            print(val)
            '''  Here, the "if statement" ensures that when we know that the iterator has run out-of-data 
            (i.e. i == 9), the iterator is re-initialized by the iterator.initializer operation '''
            if i % 9 == 0 and i > 0:
                sess.run(iterator.initializer)
                print("-----------")


# With a batch (of 3)
def simple_batch_example():
    print("\n\n*** Using a batch of 3: ")
    with tf.Session() as sess:
        sess.run(iterator_batch.initializer)
        for i in range(15):
            val = sess.run(next_element_batch)
            print(val)
            if (i + 1) % 3 == 0 and i > 0:
                sess.run(iterator_batch.initializer)
                print("-----------")



# Zipping Data
def simple_zip_example():
    print("\n\n*** Zipping Data and using a batch of 3: ")
    x = np.arange(0, 10)
    y = np.arange(1, 11)

    # Create dataset objects from the arrays
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)

    # Zip the two datasets together
    dcomb = tf.data.Dataset.zip((dx, dy)).batch(3)
    iterator_batch = dcomb.make_initializable_iterator()

    # Extract an element
    next_element_batch = iterator_batch.get_next()
    with tf.Session() as sess:
        sess.run(iterator_batch.initializer)
        for i in range(15):
            val = sess.run(next_element_batch)
            print(val)
            if (i + 1) % 3 == 0 and i > 0:
                sess.run(iterator_batch.initializer)
                print("-----------")



# Calling previous functions
if __name__ == "__main__":
    simple_example_without_batch()
    simple_batch_example()
    simple_zip_example()