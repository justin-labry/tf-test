import tensorflow as tf

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

c = a + b
sess = tf.Session()

print(sess.run(c, feed_dict={a:3, b:4.5}))
print(sess.run(c, feed_dict={a:[1,3], b:[2,4]}))