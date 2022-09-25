import numpy as np
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

X_training, Y_training = mnist.train.next_batch(5000)
X_test, Y_test = mnist.test.next_batch(200)

xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])


distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)


pred = tf.argmin(distance, 0)

accuracy = 0


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)


    for i in range(len(X_test)):

        nn_index = sess.run(pred, feed_dict={xtr: X_training, xte: X_test[i, :]})

        print("Test", i, "Prediction:", np.argmax(Y_training[nn_index]), \
            "True Class:", np.argmax(Y_test[i]))

        if np.argmax(Y_training[nn_index]) == np.argmax(Y_test[i]):
            accuracy += 1./len(X_test)
    print("Done!")
    print("Accuracy:", accuracy)