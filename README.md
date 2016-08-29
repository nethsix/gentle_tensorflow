# Gentlest Tensorflow

## Goal

Tensorflow (TF) is Google’s attempt to put the power of Deep Learning into the hands of developers around the world. It comes with a beginner & an advanced tutorial, as well as a course on Udacity. However, the materials attempt to introduce both ML and TF concurrently to solve a multi-feature problem — character recognition, which albeit interesting, unnecessarily convolutes understanding. Gentlest Tensorflow attempts to overcome that by showing how to do linear regression for a single feature problem, and expand from there.

## Cheatsheet

* cheatsheets/tensorflow_cheatsheet_1.png
  * Linear regression: single feature, single scalar outcome
  * Linear regression: multi-feature, single scalar outcome
  * Logistic regression: multi-feature, multi-class outcome

## Code

All the code are in `/code` directory:

* linear_regression_one_feature.py
  * ML with linear regression for a single feature
    * Example: predict house price from house size (single feature)
* linear_regression_one_feature_with_tensorboard.py
  * Add visualization for 'ML for single feature' with Tensorboard
    * Use tf.scalar_summary, tf.histogram_summary to collect data for variables that we want to visualize
    * Use `scope` to collapse TF network graph in to expandable/collapsible black boxes to faciliate visualization
* linear_regression_one_feature_using_mini_batch_with_tensorboard.py
  * Perform 'stochastic/mini-batch/batch' Gradient Descent with TF
  * The CUSTOMIZABLE section contains all the configurations that we can tweak, e.g., batch size, etc.
* linear_regression_multi_feature_using_mini_batch_without_matrix_with_tensorboard.py
  * ML with linear regrssion for 2 features without using 'matrix'
  * Create additional tf.Variable, tf.placeholder for each feature
  * **IMPORTANT**: This is a messy way to do ML with multiple features. This is provided as an explanation of multi-feature concept.
* linear_regression_multi_feature_using_mini_batch_with_tensorboard.py
  * ML with linear regrssion for 2 features
  * Expanding existing W (tf.Variable) in matrix 'height', and existing x (tf.placeholder) in matrix 'width' to accomodate each feature
