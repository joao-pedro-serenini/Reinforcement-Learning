import tensorflow as tf 
msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)

weights = tf.Variable(tf.random.normal([3, 2], stddev=0.1), name="weights")

# x = tf.placeholder("float", shape=None)
def my_function(x):
    y = "float"
    return y

A = tf.multiply(8, 5)
B = tf.multiply(A, 1)

tf.print(B)