import tensorflow as tf

hello = tf.constant("hello, tf!")

session = tf.Session()

print(session.run(hello))
