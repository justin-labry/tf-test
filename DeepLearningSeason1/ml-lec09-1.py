import warnings
warnings.simplefilter("ignore")

#%%

# Lab 9 XOR
import tensorflow as tf
import numpy as np

#%%
batch_size = 1000

tf.set_random_seed(777)  # for reproducibility
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


#x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
#y_data = np.array([[0], [1], [1], [1]], dtype=np.float32)

#%%
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

#X = tf.placeholder(tf.float32, [None, 2])
#Y = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.random_normal([784, 256]), name="weight")
b = tf.Variable(tf.random_normal([256]), name="bias")
layer1 = tf.nn.sigmoid(tf.matmul(X, W) + b)

W2 = tf.Variable(tf.random_normal([256, 10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')
logits = tf.sigmoid(tf.matmul(layer1, W2) + b2)

# cost/loss function
cost = tf.square(Y - logits)
#cost = -tf.reduce_mean(Y * tf.log(logits) + (1 - Y) * tf.log(1 - logits))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#%%

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, tf.cast(tf.argmax(Y,1),tf.int64)), dtype=tf.float32))


with tf.Session() as sess:

    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(batch_size):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys}
        )

        if step % 100 == 0:
            print(step, cost_val)

    # Accuracy report
    h, p, a = sess.run(
              [logits, predicted, accuracy], feed_dict={X: batch_xs, Y: batch_ys}
    )
    print("\nHypothesis: ", h, "\nCorrect: ", p, "\nAccuracy: ", a)

    expected_y, _ = sess.run([tf.argmax(logits), W2], feed_dict={X:batch_xs})
    print("expected Y", expected_y)
'''
Hypothesis:  [[ 0.5]
 [ 0.5]
 [ 0.5]
 [ 0.5]]
Correct:  [[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
Accuracy:  0.5
'''

#%%

print("hello~~~2")
