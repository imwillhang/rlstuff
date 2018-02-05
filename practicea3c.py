# A worker in A3C
# Parameters are broadcast to the worker and periodically updated

import tensorflow.contrib.slim as slim
import numpy as np

from tensorflow import nn, layers, keras, losses

# TODO:
#	make sure you get stacked frames to pass into the policy
#	make sure that you get the game to work with the policy

# Taken straight from Nature paper from DeepMind
def NatureCNN(images):
	# we apply the basic Nature feature extractor
	he_uniform = keras.initializers.he_uniform
	conv1 = layers.conv2d(images, 32, 8, 4)
	conv2 = layers.conv2d(conv1, 64, 4, 2)
	conv3 = layers.conv2d(conv2, 64, 3, 1)
	conv3 = layers.flatten(conv3)
	hOut = slim.fully_connected(conv3, 512)
	# return a 512 dimensional embedding
	return hOut

def normalizedColumnsInitializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class LSTMPolicy():
	def __init__(self, scope, reuse, inputDims, outputDims):
		self.inputs = tf.placeholder(tf.float32, [None] + inputDims)
		with tf.variable_scope(scope, reuse=reuse):
			features = NatureCNN(self.inputs)

			lstm = rnn.BasicLSTMCell(512, state_is_tuple=True)

			contextIn = tf.placeholder(tf.float32, [1, lstm.state_size.c])
			hiddenIn = tf.placeholder(tf.float32, [1, lstm.state_size.h])
			stateIn = rnn.LSTMStateTuple(contextIn, hiddenIn)

			lstmOutputs, lstmState = nn.dynamic_rnn(lstm, features, 
				initial_state=stateIn, sequence_length=)

			outputs = layers.flatten(lstmOutputs)

			self.logits = slim.fully_connected(outputs, outputDims, 
				weights_initializer=normalizedColumnsInitializer(0.01),
                biases_initializer=None)

			self.policy = nn.softmax(self.logits)

			self.value = slim.fully_connected(outputs, 1, 
				activation_fn=None,
				weights_initializer=normalizedColumnsInitializer(1.0),
                biases_initializer=None)

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

class Actor():
	def __init__(self, PolicyType, policyParams):
		self.policyFunction = PolicyType(*policyParams)

	def getActionsValues(self, state):
		actionsDistribution, values = sess.run([
			self.policyFunction.policy, self.policyFunction.value]
			feed_dict={
				self.policyFunction.inputs: state
			})
		actionsDistribution = actionsDistribution[0]
		actions = np.random.choice(actionsDistribution, p=actionsDistribution)
		actions = np.argmax(actionsDistribution == actions)
		return actions, values

class Worker():
	def __init__(self, env, name, PolicyType, policyParams, gamma, PolicyOptimizer):
		self.actor = Actor(PolicyType, policyParams)
		#self.swapPolicy = updatePolicy('global', self.name)
		self.replayBuffer = ReplayBuffer()
		self.policyOptimizer = PolicyOptimizer(
			self.actor, gamma, 'worker', 'global', 1e-4)
		self.gamma = gamma

	def runPolicy():
		with sess.as_default(), sess.graph.as_default():
			while not coord.should_stop():
				# first we need to update the worker policy with the
				#	global policy
				sess.run(self.swapPolicy)
				# start a new episode
				self.env.new_episode()
				# we need to initialize our state representation
				observation = self.env.get_state().screen_buffer
				replayBuffer.addObservation(observation)
				state = processFrame(observation)
				# rollout the policy until either
				# 	1. our replay buffer is full
				#	2. the episode is done
				# then train!
				while not self.env.is_episode_finished():
					actions, values = self.actor.getActionsValues(state)
					nextObservations, rewards, dones, infos = self.env.step(actions)
					nextState = processFrame(nextObservations)
					replayBuffer.addTransition(state, action, reward, done, nextState, values)
					state = nextState
					# if our buffer is full
					if self.replayBuffer.size >= self.batchSize:
						# process rollouts first to calculate discounted rewards
						bootstrap = None
						if done:
							_, bootstrap = self.actor.getActionsValues(state)
						self.trainPolicy(bootstrapVal=bootstrap)
			# train after rollout!
			if self.replayBuffer.size > 0:
				self.trainPolicy()

	def trainPolicy(self, bootstrapVal=None):
		# handles getting the appropriate trajectories and 
		# performing backups using policy gradient
		self.replayBuffer.processRollouts(
			bootstrapVal=bootstrapVal, gamma=self.gamma)
		# perform training
		sess.run([
			self.optimizer.policyGradientLoss,
			self.optimizer.valueFunctionLoss,
			self.optimizer.entropy,
			self.optimizer.train],
			feed_dict={
				self.optimizer.actions: self.replayBuffer.actions,
				self.optimizer.values: self.replayBuffer.discountedRewards,
				self.optimizer.advantages: self.replayBuffer.advantages,
				self.optimizer.actor.inputs: self.replayBuffer.states
			})
		# empty the replay buffer so we can store stuff in it again
		self.replayBuffer.empty()

def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class ReplayBuffer():
	def __init__(self):
		self.empty()

	def processRollouts(self, bootstrapVal=bootstrapVal, gamma=gamma):
		# add bootstrap value to rewards
		rewards_ = self.rewards + [bootstrapVal]
		self.discountedRewards = discount(self.rewards_, gamma)[:-1]
		values_ = self.values + [bootstrapVal]
		self.advantages = rewards + gamma * self.values_[1:] - self.values_[:-1]
		self.advantages = discount(self.advantages, gamma)

	def addTransition(self, state, action, reward, done, nextState, value):
		self.size += 1
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.dones.append(done)
		self.nextState.append(nextState)
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
	def __init__(self, actor, gamma, scope, globalScope, lr):
		self.actor = actor
		self.gamma = gamma
		self.scope = scope
		self.globalScope = globalScope
		self.lr = lr
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

		negativeLogLoss = nn.sparse_softmax_cross_entropy_with_logits(
			logits=self.actor.logits, labels=self.actions)

		self.policyGradientLoss = tf.reduce_mean(self.advantages * negativeLogLoss)
		self.valueFunctionLoss = losses.mean_squared_error(self.values, self.actor.value)

		self.entropy = tf.reduce_mean(self.policy * tf.log(self.policy))
		# the key is to drive vFLoss towards 0, pGLoss to a large negative number, 
		# 	and entropy to 0 to minimize loss.
		self.loss = valueLossCoeff * valueFunctionLoss + \
					policyGradientLoss + \
					entropyCoeff * entropy
		variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
		self.gradients = tf.gradients(loss, variables)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
		if maxGradNorm is not None:
			self.gradients, norm = tf.clip_by_global_norm(gradients, maxGradNorm)

	def updateGlobalVariables(self):
		# performs a global update
		# if we use MPI we can broadcast the gradients to the param server
		globalVariables = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, self.globalScope)
		globalGradsAndVars = zip(self.gradients, globalVariables)
		self.train = optimizer.apply_gradients(globalGradsAndVars)

	def updateLocalVariables(self):
		# performs a local update
		localVariables = tf.get_collection(
			tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
		gradsAndVars = zip(self.gradients, localVariables)
		self.train = optimizer.apply_gradients(gradsAndVars)
