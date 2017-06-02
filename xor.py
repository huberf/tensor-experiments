import tensorflow as tf
import random

training_data = [[0, 1], [1, 0], [1, 1], [0, 0]]

a = tf.placeholder(tf.float32, [None])
b = tf.placeholder(tf.float32, [None])

diff = tf.Variable(1.1, name="Summer")

y = tf.nn.softmax(a + b)

y_ = tf.placeholder(tf.int32, [None])

difference = tf.reduce_mean( a + b - diff,  y_)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(difference) #TODO

sess = tf.InteractiveSession()
print(tf.trainable_variables())
tf.global_variables_initializer().run()


for i in range(1000):
    z = training_data[random.randint(0, 3)]
    yes = 0
    if sum(z) == 1:
        yes = 1
    sess.run(train_step, feed_dict={a: z[0], b: z[1], y_: yes})
