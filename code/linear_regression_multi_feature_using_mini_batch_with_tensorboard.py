import numpy as np
import tensorflow as tf

# CUSTOMIZABLE: Collect/Prepare data
datapoint_size = 1000
batch_size = 1000
steps = 10000
actual_W1 = 2
actual_W2 = 5
actual_b = 7 
learn_rate = 0.001
log_file = "/tmp/feature_2_batch_1000"

# Model linear regression y = Wx + b
x = tf.placeholder(tf.float32, [None, 2], name="x")
W = tf.Variable(tf.zeros([2,1]), name="W")
b = tf.Variable(tf.zeros([1]), name="b")
with tf.name_scope("Wx_b") as scope:
  product = tf.matmul(x,W)
  y = product + b

# Add summary ops to collect data
W_hist = tf.histogram_summary("weights", W)
b_hist = tf.histogram_summary("biases", b)
y_hist = tf.histogram_summary("y", y)

y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_-y))
  cost_sum = tf.scalar_summary("cost", cost)

# Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

all_xs = []
all_ys = []
for i in range(datapoint_size):
  # Create fake data for y = 2.x_1 + 5.x_2 + 7
  x_1 = i%10
  x_2 = np.random.randint(datapoint_size/2)%10
  y = actual_W1 * x_1 + actual_W2 * x_2 + actual_b
  # Create fake data for y = W.x + b where W = [2, 5], b = 7
  all_xs.append([x_1, x_2])
  all_ys.append(y)

all_xs = np.array(all_xs)
all_ys = np.transpose([all_ys])

sess = tf.Session()

# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.merge_all_summaries()
writer = tf.train.SummaryWriter(log_file, sess.graph_def)

init = tf.initialize_all_variables()
sess.run(init)

for i in range(steps):
  if datapoint_size == batch_size:
    batch_start_idx = 0
  elif datapoint_size < batch_size:
    raise ValueError("datapoint_size: %d, must be greater than batch_size: %d" % (datapoint_size, batch_size))
  else:
    batch_start_idx = (i * batch_size) % (datapoint_size - batch_size)
  batch_end_idx = batch_start_idx + batch_size
  batch_xs = all_xs[batch_start_idx:batch_end_idx]
  batch_ys = all_ys[batch_start_idx:batch_end_idx]
  xs = np.array(batch_xs)
  ys = np.array(batch_ys)
  all_feed = { x: all_xs, y_: all_ys }
  # Record summary data, and the accuracy every 10 steps
  if i % 10 == 0:
    result = sess.run(merged, feed_dict=all_feed)
    writer.add_summary(result, i)
  else:
    feed = { x: xs, y_: ys }
    sess.run(train_step, feed_dict=feed)
  print("After %d iteration:" % i)
  print("W: %s" % sess.run(W))
  print("b: %f" % sess.run(b))
  print("cost: %f" % sess.run(cost, feed_dict=all_feed))

# NOTE: W should be close to actual_W1, actual_W2, and b should be close to actual_b
