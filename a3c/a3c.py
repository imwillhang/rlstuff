import tensorflow as tf
import numpy as np
import scipy.signal
import queue
import matplotlib.pyplot as plt
import threading

import cv2

from models import CNNPolicy, LinearPolicy
from tensorflow import losses

def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class Rollout():
	def __init__(self):
		self.size = 0
		self.bootstrap = 0
		self.total_reward = 0
		self.episode = None
		self.iteration = None
		self.done = False
		self.states = []
		self.actions = []
		self.rewards = []
		self.next_states = []
		self.values = []

	def add_transition(self, state, action, reward, done, next_state, value):
		self.size += 1
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.next_states.append(next_state)
		self.values.append(value)

class Actor():
	def __init__(self, config, sess, scope):
		self.config = config
		self.sess = sess
		self.num_actions = config.output_dims
		if config.policy_type == 'cnn':
			self.policy = CNNPolicy(config, scope)
		elif config.policy_type == 'linear':
			self.policy = LinearPolicy(config, scope)

	def act(self, state):
		actions, values, logits, pi = self.sess.run([
			self.policy.actions, self.policy.vf, self.policy.logits, self.policy.pi],
			feed_dict={
				self.policy.inputs: state
			})
		# action_dist = action_dist[0]
		# actions = np.random.choice(list(range(self.num_actions)), 
		# 	p=action_dist)
		#print(logits)
		return [actions], values

class Worker():
	def __init__(self, 
		env, 
		name,
		sess,
		output_path, 
		config):

		self.env = env
		self.num_envs = env.num_envs
		self.name = name
		self.sess = sess
		self.config = config
		self.output_path = output_path

		self.total_iterations = 0
		self.episodes = 0

		worker_device = "/job:worker/task:{}/device:cpu:0".format(config.task_index)
		#with tf.device(tf.train.replica_device_setter(1, worker_device=worker_device)):
		with tf.device("/job:ps/task:0/device:cpu:0"):
			#with tf.variable_scope("global"):
			self.global_actor = Actor(config, sess, 'global')
		with tf.device(worker_device):
			#with tf.variable_scope("local"):
			self.actor = Actor(config, sess, 'local')

		self.rollouts = queue.Queue(config.max_rollouts)

		self.accum_rewards = 0

		self._sync = tf.group(*[v1.assign(v2) for v1, v2 in zip(self.actor.policy.vars, self.global_actor.policy.vars)])

		self.a3c()
		self.add_summary()

	def run(self):
		threading.Thread(target=self._run).start()

	def _run(self):
		rollout = self.rollout()
		while True:
			self.rollouts.put(next(rollout), timeout=600.)

	def rollout(self):
		# normalize the frame values to [0, 1] and housekeeping
		state = self.env.reset().astype(np.float32) / 255.
		rollout = Rollout()
		total_reward = 0
		while True:
			self.total_iterations += 1
			action, value = self.actor.act(state)
			next_state, reward, done, info = self.env.step(action)
			next_state = next_state.astype(np.float32) / 255.
			# add this transition to the replay buffer!
			reward = reward[-1]
			rollout.add_transition(state, action, reward, done, next_state, value)
			# update the state
			state = next_state
			# housekeeping
			done = done[-1]
			total_reward += reward
			# if our buffer is full, we yield a new rollout, and then continue playing the same game
			if rollout.size >= self.config.batch_size or done:
				# process rollouts first to calculate discounted rewards
				bootstrap = 0
				# if we aren't finished with the game, we need to bootstrap by calculating V(s')
				if not done:
					_, bootstrap = self.actor.act(next_state)
				# if we're finished, our final value of V(s_term) will be 0
				else:
					rollout.total_reward = total_reward
					rollout.done = True
					rollout.episode = self.episodes
					rollout.iteration = self.total_iterations
					total_reward = 0
					self.episodes += 1
				rollout.bootstrap = bootstrap
				yield rollout
				# make a new rollout!
				rollout = Rollout()

	def train(self, bootstrap=None):
		# get a rollout
		rollout = self.rollouts.get(timeout=600.)
		# and process it!
		target_v, adv = self.process_rollouts(rollout)
		# then run the results through the computation graph
		_, summary, entropy, loss, logits, v = self.sess.run(
			[self.train_op, self.merged, self.entropy, self.loss, self.actor.policy.logits, self.target_v],
			feed_dict={
				self.action: np.reshape(rollout.actions, (-1)),
				self.adv: adv,
				self.target_v: target_v,
				self.actor.policy.inputs: np.reshape(rollout.states, (-1,) + tuple(self.config.input_dims))
			})

		self.accum_rewards += rollout.total_reward

		if rollout.done:
			print('==============================')
			print('Loss/Entropy for episode {} is {}/{}'.format(rollout.episode, loss, entropy))
			print('Reward for episode {} is {}'.format(rollout.episode, self.accum_rewards))
			print('Now at iteration {}'.format(rollout.iteration))

			reward_summary = tf.Summary()
			reward_summary.value.add(tag='Episode Rewards', simple_value=self.accum_rewards)
			self.file_writer.add_summary(reward_summary, rollout.iteration)
			self.file_writer.add_summary(summary, rollout.iteration)
			self.file_writer.flush()

			self.accum_rewards = 0

	def add_summary(self):
		tf.summary.scalar("Loss", self.loss)
		tf.summary.scalar("Entropy", self.entropy)
		self.merged = tf.summary.merge_all()
		self.file_writer = tf.summary.FileWriter(self.output_path, self.sess.graph)

	def a3c(self):
		self.action = tf.placeholder(tf.int32, shape=[None])
		self.adv = tf.placeholder(tf.float32, shape=[None])
		self.target_v = tf.placeholder(tf.float32, shape=[None])

		actor = self.actor
		# make sure we pass in logits
		log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.action,
			logits=actor.policy.logits)

		policy_loss = tf.reduce_mean(self.adv * log_prob)
		value_loss = tf.losses.mean_squared_error(self.target_v, actor.policy.vf)
		# make sure we pass in the probability distribution
		self.entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(actor.policy.logits) * \
			tf.nn.log_softmax(actor.policy.logits + 1e-7), axis=1))

		self.loss = policy_loss + self.config.vf_coeff * value_loss - \
			self.config.entropy_coeff * self.entropy

		variables = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'local'), key=lambda x: x.name[0])
		global_variables = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'), key=lambda x: x.name[0])
		gradients = tf.gradients(self.loss, variables)
		optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
		gradients, _ = tf.clip_by_global_norm(gradients, self.config.max_grad_norm)

		grads_and_vars = zip(gradients, global_variables)
		self.train_op = optimizer.apply_gradients(grads_and_vars)

	def process_rollouts(self, rollout):
		# if there is a valid bootstrap value, we use it in order to get a proper estimate of V
		reward_with_v = np.concatenate([rollout.rewards, [0 if rollout.bootstrap == None else rollout.bootstrap]])
		value_with_v = np.concatenate([rollout.values, [0 if rollout.bootstrap == None else rollout.bootstrap]])
		# we get the discounted rewards
		disc_reward = discount(reward_with_v, self.config.gamma)[:-1]
		# get the temporal difference
		temporal_diff = rollout.rewards + self.config.gamma * value_with_v[1:] - value_with_v[:-1]
		# this is the generalized advantage estimator
		advantage = discount(temporal_diff, self.config.gamma)
		return disc_reward, advantage

	def sync(self):
		self.sess.run([self._sync])

	def update(self):
		pass