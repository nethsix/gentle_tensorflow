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
log_file = "/tmp/feature_2_batch_1000_without_matrix"

# Model linear regression y = Wx + b
x1 = tf.placeholder(tf.float32, [None, 1], name="x1")
x2 = tf.placeholder(tf.float32, [None, 1], name="x2")
W1 = tf.Variable(tf.zeros([1,1]), name="W1")
W2 = tf.Variable(tf.zeros([1,1]), name="W2")
b = tf.Variable(tf.zeros([1]), name="b")
with tf.name_scope("Wx_b") as scope:
  product_1 = tf.matmul(x1,W1)
  product_2 = tf.matmul(x2,W2)
  y = product_1 + product_2 + b

# Add summary ops to collect data
W1_hist = tf.summary.histogram("weights_1", W1)
W2_hist = tf.summary.histogram("weights_2", W2)
b_hist = tf.summary.histogram("biases", b)
y_hist = tf.summary.histogram("y", y)

y_ = tf.placeholder(tf.float32, [None, 1])

# Cost function sum((y_-y)**2)
with tf.name_scope("cost") as scope:
  cost = tf.reduce_mean(tf.square(y_-y))
  cost_sum = tf.summary.scalar("cost", cost)

# Training using Gradient Descent to minimize cost
with tf.name_scope("train") as scope:
  train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(cost)

all_x1s = []
all_x2s = []
all_ys = []
for i in range(datapoint_size):
  # Create fake data for y = 2.x_1 + 5.x_2 + 7
  x_1 = i%10
  x_2 = np.random.randint(datapoint_size/2)%10
  y = actual_W1 * x_1 + actual_W2 * x_2 + actual_b
  # Create fake data for y = W.x + b where W = [2, 5], b = 7
  all_x1s.append([x_1])
  all_x2s.append([x_2])
  all_ys.append(y)

all_x1s = np.array(all_x1s)
all_x2s = np.array(all_x2s)
all_ys = np.transpose([all_ys])

sess = tf.Session()

# Merge all the summaries and write them out to /tmp/mnist_logs
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_file, sess.graph_def)

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
  batch_x1s = all_x1s[batch_start_idx:batch_end_idx]
  batch_x2s = all_x2s[batch_start_idx:batch_end_idx]
  batch_ys = all_ys[batch_start_idx:batch_end_idx]
  x1s = np.array(batch_x1s)
  x2s = np.array(batch_x2s)
  ys = np.array(batch_ys)
  all_feed = { x1: all_x1s, x2: all_x2s, y_: all_ys }
  # Record summary data, and the accuracy every 10 steps
  if i % 10 == 0:
    result = sess.run(merged, feed_dict=all_feed)
    writer.add_summary(result, i)
  else:
    feed = { x1: x1s, x2: x2s, y_: ys }
    sess.run(train_step, feed_dict=feed)
  print("After %d iteration:" % i)
  print("W1: %s" % sess.run(W1))
  print("W2: %s" % sess.run(W2))
  print("b: %f" % sess.run(b))
  print("cost: %f" % sess.run(cost, feed_dict=all_feed))

# NOTE: W1, W2 should be close to actual_W1, actual_W2, and b should be close to actual_b
