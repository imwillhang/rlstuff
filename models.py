import tensorflow.contrib.layers as layers
import tensorflow as tf
import numpy as np

from tensorflow import nn

def NatureCNN(images):
	# we apply the basic Nature feature extractor
	conv1_1 = layers.conv2d(images, 32, 8, stride=4)
	conv2_1 = layers.conv2d(conv1_1, 64, 4, stride=2)
	conv3_1 = layers.conv2d(conv2_1, 64, 3, stride=1)
	conv3_1 = layers.flatten(conv3_1)
	hidden_1 = layers.fully_connected(conv3_1, 512)
	# return a 512 dimensional embedding
	return hidden_1

def normalizedColumnsInitializer(std=1.0):
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

class LinearPolicy(object):
	def __init__(self, config):
		self.inputs = tf.placeholder(tf.float32, shape=[None] + config.input_dims)
		with tf.variable_scope(config.scope):
			hidden1 = layers.fully_connected(self.inputs, 18)
			hidden2 = layers.fully_connected(hidden1, 36)
			hidden3 = layers.fully_connected(hidden1, 18)
			self.logits = layers.fully_connected(hidden3, config.output_dims,
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(0.01),
				biases_initializer=None)
			self.pi = nn.softmax(self.logits)
			self.actions = tf.squeeze(tf.multinomial(self.logits, 1))
			self.vf = layers.fully_connected(hidden3, 1, 
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(1.0),
				biases_initializer=None)
			# we want self.value to have shape (None,) when it is still (None, 1) right now
			self.vf = tf.squeeze(self.vf)

class CNNPolicy():
	def __init__(self, config):
		self.inputs = tf.placeholder(tf.float32, shape=[None] + config.input_dims)
		with tf.variable_scope(config.scope):
			conv1_1 = layers.conv2d(self.inputs, 32, 8, stride=4)
			conv2_1 = layers.conv2d(conv1_1, 64, 4, stride=2)
			conv3_1 = layers.conv2d(conv2_1, 64, 3, stride=1)
			conv3_1 = layers.flatten(conv3_1)
			features = layers.fully_connected(conv3_1, 512)
			features = layers.flatten(features)
			features = layers.fully_connected(features, 256,
				weights_initializer=normalizedColumnsInitializer(0.01),
				biases_initializer=None)
			self.logits = layers.fully_connected(features, config.output_dims,
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(0.01),
				biases_initializer=None)
			self.pi = nn.softmax(self.logits)
			self.actions = tf.squeeze(tf.multinomial(self.logits, 1))
			self.vf = layers.fully_connected(features, 1, 
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(1.0),
				biases_initializer=None)
			# we want self.value to have shape (None,) when it is still (None, 1) right now
			self.vf = tf.squeeze(self.vf)

class DuelDQN():
	def __init__(self, config):
		self.inputs = tf.placeholder(tf.float32, shape=[None] + config.input_dims)
		with tf.variable_scope(config.scope, reuse=config.reuse):
			conv1_1 = layers.conv2d(self.inputs, 32, 8, stride=4)
			conv2_1 = layers.conv2d(conv1_1, 64, 4, stride=2)
			conv3_1 = layers.conv2d(conv2_1, 64, 3, stride=1)
			conv3_1 = layers.flatten(conv3_1)
			hidden_v = layers.fully_connected(conv3_1, 512)
			hidden_a = layers.fully_connected(conv3_1, 512)
			value = layers.fully_connected(hidden_v, 1, activation_fn=None)
			advantage = layers.fully_connected(hidden_a, num_actions, activation_fn=None)
			self.q_func = value + advantage - tf.reduce_mean(advantage)