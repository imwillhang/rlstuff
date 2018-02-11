# A worker in A3C
# Parameters are broadcast to the worker and periodically updated

import tensorflow as tf

import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim

import numpy as np
from preprocess import greyscale as processFrame

from tensorflow import nn, losses

import scipy.signal

sess = tf.Session()

# Taken straight from Nature paper from DeepMind
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

# class LSTMPolicy():
# 	def __init__(self, scope, reuse, inputDims, outputDims):
# 		self.inputs = tf.placeholder(tf.float32, [None] + inputDims)
# 		with tf.variable_scope(scope, reuse=reuse):
# 			features = NatureCNN(self.inputs)

# 			lstm = rnn.BasicLSTMCell(512, state_is_tuple=True)

# 			contextIn = tf.placeholder(tf.float32, [1, lstm.state_size.c])
# 			hiddenIn = tf.placeholder(tf.float32, [1, lstm.state_size.h])
# 			stateIn = rnn.LSTMStateTuple(contextIn, hiddenIn)

# 			lstmOutputs, lstmState = nn.dynamic_rnn(lstm, features, 
# 				initial_state=stateIn, sequence_length=)

# 			outputs = layers.flatten(lstmOutputs)

# 			self.logits = slim.fully_connected(outputs, outputDims, 
# 				weights_initializer=normalizedColumnsInitializer(0.01),
#                 biases_initializer=None)

# 			self.policy = nn.softmax(self.logits)

# 			self.value = slim.fully_connected(outputs, 1, 
# 				activation_fn=None,
# 				weights_initializer=normalizedColumnsInitializer(1.0),
#                 biases_initializer=None)

class CNNPolicy():
	def __init__(self, scope, reuse, inputDims, outputDims):
		self.inputs = tf.placeholder(tf.float32, [None] + inputDims)
		with tf.variable_scope(scope, reuse=reuse):
			features = NatureCNN(self.inputs)
			outputs = layers.flatten(features)
			self.logits = slim.fully_connected(outputs, outputDims, 
				weights_initializer=normalizedColumnsInitializer(0.01),
                biases_initializer=None)
			self.policy = nn.softmax(self.logits)
			self.value = slim.fully_connected(outputs, 1, 
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(1.0),
                biases_initializer=None)
			# we want self.value to have shape (None,) when it is still (None, 1) right now
			self.value = tf.squeeze(self.value)

class Actor():
	def __init__(self, PolicyType, policyParams):
		self.numActions = policyParams['outputDims']
		self.policyFunction = PolicyType(**policyParams)

	def getActionsValues(self, state):
		actionsDistribution, values = sess.run(
			[self.policyFunction.policy, self.policyFunction.value],
			feed_dict={
				self.policyFunction.inputs: state
			})
		actionsDistribution = actionsDistribution[0]
		actions = np.random.choice(list(range(self.numActions)), 
			p=actionsDistribution)
		# actions = np.argmax(actionsDistribution)
		return actions, values

class Worker():
	def __init__(self, env, name, PolicyType, policyParams, gamma, 
		PolicyOptimizer, outputPath, maxGradNorm=0.5, history=4, batchSize=64):
		self.gamma = gamma
		self.history = history
		self.batchSize = batchSize
		self.totalTransitions = 0
		self.outputPath = outputPath
		self.env = env
		self.gamma = gamma
		self.history = history
		self.batchSize = batchSize
		self.totalTransitions = 0
		self.outputPath = outputPath
		
		self.actor = Actor(PolicyType, policyParams)
		#remove#self.swapPolicy = updatePolicy('global', self.name)
		self.replayBuffer = ReplayBuffer(policyParams['inputDims'])
		self.policyOptimizer = PolicyOptimizer(
			self.actor, gamma, 'worker', None, 5e-4, maxGradNorm)
		self.addSummary()

	def runPolicy(self):
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		with sess.as_default(), sess.graph.as_default():
			#while not coord.should_stop():
			# first we need to update the worker policy with the
			#	global policy
			#remove#sess.run(self.swapPolicy)
			# start a new episode
			frame = self.env.reset()
			# we need to initialize our state representation
			frame = np.expand_dims(processFrame(frame), axis=0)
			# we need to stick four frames together in the beginning
			frameBuffer = [frame] * self.history
			state = np.concatenate(frameBuffer, axis=3)
			# rollout the policy until either
			# 	1. our replay buffer is full
			#	2. the episode is done
			# then train!
			while True:
				self.totalTransitions += 1
				action, value = self.actor.getActionsValues(state)
				nextFrame, reward, done, info = self.env.step(action)

				nextFrame = np.expand_dims(processFrame(nextFrame), axis=0)
				frameBuffer = frameBuffer[:-1] + [nextFrame]
				nextState = np.concatenate(frameBuffer, axis=3)
				
				self.replayBuffer.addTransition(state, action, reward, done, nextState, value)
				state = nextState

				#self.env.render()
				# if our buffer is full
				if self.replayBuffer.size >= self.batchSize:
					# process rollouts first to calculate discounted rewards
					bootstrap = None
					if done:
						_, bootstrap = self.actor.getActionsValues(state)
					self.trainPolicy(bootstrapVal=bootstrap)

				if done:
					break
		# train after rollout!
		if self.replayBuffer.size > 0:
			self.trainPolicy()

	def trainPolicy(self, bootstrapVal=None):
		# handles getting the appropriate trajectories and 
		# performing backups using policy gradient
		self.replayBuffer.processRollouts(
			bootstrapVal=bootstrapVal, gamma=self.gamma)
		# perform training
		_, summary = sess.run([
			self.policyOptimizer.train, self.merged],
			feed_dict={
				self.policyOptimizer.actions: self.replayBuffer.actions,
				self.policyOptimizer.values: self.replayBuffer.discountedRewards,
				self.policyOptimizer.advantages: self.replayBuffer.advantages,
				self.policyOptimizer.rewards: self.replayBuffer.rewards,
				self.actor.policyFunction.inputs: self.replayBuffer.states
			})
		self.file_writer.add_summary(summary, self.totalTransitions)

		print('At iteration {}, our mean reward is {}'.format(self.totalTransitions, self.replayBuffer.meanRewards))
		# empty the replay buffer so we can store stuff in it again
		self.replayBuffer.empty()

	def addSummary(self):
		tf.summary.scalar("Mean Reward", self.policyOptimizer.meanRewards)
		tf.summary.scalar("PG Loss", self.policyOptimizer.policyGradientLoss)
		tf.summary.scalar("VF Loss", self.policyOptimizer.valueFunctionLoss)

		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.outputPath, sess.graph)

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class ReplayBuffer():
	def __init__(self, stateDims):
		self.empty()
		self.stateDims = stateDims

	def processRollouts(self, bootstrapVal=None, gamma=0.99):
		# add bootstrap value to rewards
		if bootstrapVal == None:
			bootstrapVal = 0
		rewards_ = np.array(self.rewards + [bootstrapVal])
		self.discountedRewards = discount(rewards_, gamma)[:-1]
		values_ = np.array(self.values + [bootstrapVal])
		self.advantages = self.rewards + gamma * values_[1:] - values_[:-1]
		self.advantages = discount(self.advantages, gamma)

		#self.states = np.squeeze(self.states, axis=1)
		self.states = np.reshape(self.states, (-1,) + tuple(self.stateDims))
		self.meanRewards = np.mean(self.rewards)

	def addTransition(self, state, action, reward, done, nextState, value):
		self.size += 1
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.dones.append(done)
		self.nextStates.append(nextState)
		self.values.append(value)

	def empty(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.nextStates = []
		self.values = []
		self.size = 0

class GAEOptimizer():
	def __init__(self, actor, gamma, scope, globalScope, lr, maxGradNorm):
		self.actor = actor
		self.gamma = gamma
		self.scope = scope
		self.globalScope = globalScope
		self.lr = lr
		self.maxGradNorm = maxGradNorm
		self.policyGradient()
		if self.globalScope != None:
			self.updateGlobalVariables()
		else:
			self.updateLocalVariables()

	def policyGradient(self, valueLossCoeff=0.5, entropyCoeff=0.01):
		# takes raw trajectory and rollout information and calculates gradients
		self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
		self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)
		self.values = tf.placeholder(shape=[None], dtype=tf.float32)
		self.rewards = tf.placeholder(shape=[None], dtype=tf.float32)

		self.meanRewards = tf.reduce_mean(self.rewards)

		negativeLogLoss = nn.sparse_softmax_cross_entropy_with_logits(
			logits=self.actor.policyFunction.logits, labels=self.actions)

		self.policyGradientLoss = tf.reduce_mean(self.advantages * negativeLogLoss)
		self.valueFunctionLoss = losses.mean_squared_error(self.values, self.actor.policyFunction.value)

		self.entropy = tf.reduce_mean(self.actor.policyFunction.policy * tf.log(self.actor.policyFunction.policy))
		# the key is to drive vFLoss towards 0, pGLoss to a large negative number, 
		# 	and entropy to 0 to minimize loss.
		self.loss = valueLossCoeff * self.valueFunctionLoss + \
					self.policyGradientLoss + \
					entropyCoeff * self.entropy
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		self.gradients = tf.gradients(self.loss, variables)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		if self.maxGradNorm is not None:
			self.gradients, norm = tf.clip_by_global_norm(self.gradients, self.maxGradNorm)

	def updateGlobalVariables(self):
		# performs a global update
		# if we use MPI we can broadcast the gradients to the param server
		globalVariables = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, self.globalScope)
		globalGradsAndVars = zip(self.gradients, globalVariables)
		self.train = self.optimizer.apply_gradients(globalGradsAndVars)

	def updateLocalVariables(self):
		# performs a local update
		localVariables = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		gradsAndVars = zip(self.gradients, localVariables)
		self.train = self.optimizer.apply_gradients(gradsAndVars)
