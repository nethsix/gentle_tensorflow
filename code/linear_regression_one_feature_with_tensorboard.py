import numpy as np
import tensorflow as tf

# Model linear regression y = Wx + b
x = tf.placeholder(tf.float32, [None, 1], name="x-input")
W = tf.Variable(tf.zeros([1,1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
with tf.name_scope("Wx_b") as scope:
  product = tf.matmul(x,W)
  y = product + b

# Add summary ops to collect data
W_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

y_ = tf.placeholder(tf.float32, [None, 1], name="y-input")

# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_-y))
  cost_sum = tf.scalar_summary("cost", cost)

# Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

sess = tf.Session()

# Merge all the summaries and write them out to logfile
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter("/tmp/mnist_sgd_logs_20160404", sess.graph_def)

init = tf.initialize_all_variables()
sess.run(init)

steps = 1000

# Train
for i in range(steps):
  # Create fake data for y = W.x + b where W = 2, b = 0
  xs = np.array([[i]])
  ys = np.array([[2*i]])
  feed = { x: xs, y_: ys }
  sess.run(train_step, feed_dict=feed)
  print("After %d iteration:" % i)
  print("W: %f" % sess.run(W))
  print("b: %f" % sess.run(b))
  # Record summary data, and the accuracy every 10 steps
  if i % 10 == 0:
    result = sess.run(merged, feed_dict=feed)
    writer.add_summary(result, i)

# NOTE: W should be close to 2, and b should be close to 0
