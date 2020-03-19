import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.constant(7.1)
node4 = tf.add_n([node1, node2, node3])

print("node1:", node1, "node2:",node2)
print("node4:", node4)

session = tf.Session()
print("sess.run(node1, node2...):", session.run([node1,node2,node3]))
print("session.ruN(node4):", session.run(node4))