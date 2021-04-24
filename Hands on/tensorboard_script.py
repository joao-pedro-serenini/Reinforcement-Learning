import tensorflow as tf

a = tf.constant(5)
b = tf.constant(4)
c = tf.multiply(a, b)
d = tf.constant(2)
e = tf.constant(3)
f = tf.multiply(d, e)
g = tf.add(c, f)

writer = tf.summary.create_file_writer("output")
print(g)
writer.close()