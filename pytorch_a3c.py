import torch
import torch.nn.functional as F
import torch.optim as optim

import tensorflow as tf

from torch.autograd import Variable
from torch import nn

import numpy as np
import scipy.signal

from preprocess import greyscale as process_frame
from pytorch_models import CNNPolicy
import matplotlib.pyplot as plt

import cv2

def discount(x, gamma):
	return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class ReplayBuffer():
	def __init__(self, state_dims):
		self.empty()
		self.state_dims = state_dims

	def add_transition(self, state, action, reward, done, next_state, value):
		self.size += 1
		self.states.append(state)
		self.actions.append(action)
		self.rewards.append(reward)
		self.dones.append(done)
		self.next_states.append(next_state)
		self.values.append(value)

	def empty(self):
		self.states = []
		self.actions = []
		self.rewards = []
		self.dones = []
		self.next_states = []
		self.values = []
		self.size = 0

class Actor():
	def __init__(self, config, sess):
		self.config = config
		self.sess = sess
		self.num_actions = config.output_dims
		if config.policy_type == 'cnn':
			self.policy = CNNPolicy(config)

	def act(self, state):
		logits, values = self.eval(state)
		pi = F.softmax(logits, dim=1)
		actions = torch.multinomial(pi).data.numpy()[0]
		values = values.data.numpy()[0][0]
		return actions, values

	def eval(self, state):
		state = Variable(torch.FloatTensor(state))
		state = state.permute(0, 3, 1, 2)
		return self.policy(state)

class Worker():
	def __init__(self, 
		env, 
		name,
		sess,
		output_path, config):

		self.env = env
		self.name = name
		self.sess = sess
		self.config = config

		self.total_iterations = 0
		self.episodes = 0

		self.actor = Actor(config, sess)
		self.replay_buffer = ReplayBuffer(config.input_dims)

		self.actor.policy.eval()
		self.optimizer = optim.Adam(self.actor.policy.parameters(), lr=self.config.lr)
		self.file_writer = tf.summary.FileWriter(output_path, tf.Session().graph)

	def process_state(self, frame, buff):
		frame = np.expand_dims(process_frame(frame), axis=0)
		buff = buff[1:] + [frame]
		state = np.concatenate(buff, axis=3)
		#show_images([np.squeeze(img) for img in buff])
		return state, buff

	def run(self):
		frame = self.env.reset()
		frame_ = np.expand_dims(process_frame(frame), axis=0)
		buff = [frame_] * self.config.history
		state, buff = self.process_state(frame, buff)

		total_reward = 0
		while True:
			self.total_iterations += 1
			action, value = self.actor.act(state)
			next_frame, reward, done, info = self.env.step(action)

			next_state, buff = self.process_state(next_frame, buff)
			
			self.replay_buffer.add_transition(state, action, reward, done, next_state, value)
			state = next_state

			self.env.render()

			total_reward += reward
			# if our buffer is full
			if self.replay_buffer.size >= self.config.batch_size:
				# process rollouts first to calculate discounted rewards
				bootstrap = None

				if not done:
					_, bootstrap = self.actor.act(state)

				self.train(bootstrap=bootstrap)

			if done:
				break
		# train after rollout!
		if self.replay_buffer.size > 0:
			self.train()

		self.episodes += 1
		#if self.episodes % 10 == 0:
		print('==============PYTORCH================')
		print('Loss/Entropy for episode {} is {}/{}'.format(self.episodes, self.report_loss, self.report_entropy))
		print('Reward for episode {} is {}'.format(self.episodes, total_reward))
		print('Now at iteration {}'.format(self.total_iterations))

		summary = tf.Summary()
		summary.value.add(tag='Episode Rewards', simple_value=total_reward)
		summary.value.add(tag='Entropy', simple_value=self.report_entropy)
		self.file_writer.add_summary(summary, self.total_iterations)
		self.file_writer.flush()

	def train(self, bootstrap=None):
		self.actor.policy.train()
		target_v, adv = self.process_rollouts(bootstrap=bootstrap)
		self.replay_buffer.states = np.reshape(self.replay_buffer.states, (-1,) + tuple(self.replay_buffer.state_dims))
		loss, entropy = self.a3c(target_v, adv)
		self.report_loss = loss
		self.report_entropy = entropy
		self.replay_buffer.empty()
		self.actor.policy.eval()

	def a3c(self, target_v, adv):
		#states = Variable(torch.from_numpy(np.array(self.replay_buffer.states)))
		states = np.array(self.replay_buffer.states)
		actions = Variable(torch.LongTensor(np.array(self.replay_buffer.actions)))
		target_v = Variable(torch.FloatTensor(np.array(target_v)))
		adv = Variable(torch.FloatTensor(np.array(adv)))

		logits, vf = self.actor.eval(states)
		probs = F.softmax(logits, dim=1)
		log_probs = F.log_softmax(logits, dim=1)

		entropy = -torch.mean(torch.sum(log_probs * probs, 1))
		policy_loss = -torch.mean(log_probs.gather(1, actions) * adv)
		value_loss = torch.mean(torch.pow(target_v - vf, 2))
		loss = policy_loss + value_loss * self.config.vf_coeff - entropy * self.config.entropy_coeff
		
		self.optimizer.zero_grad()
		loss.backward()
		if self.config.max_grad_norm: 
			nn.utils.clip_grad_norm(self.actor.policy.parameters(), self.config.max_grad_norm)
		self.optimizer.step()

		return loss.data.numpy()[0], entropy.data.numpy()[0]

	def process_rollouts(self, bootstrap=None):
		buff = self.replay_buffer
		reward_with_v = np.array(buff.rewards + [0 if bootstrap == None else bootstrap])
		value_with_v = np.array(buff.values + [0 if bootstrap == None else bootstrap])
		disc_reward = discount(reward_with_v, self.config.gamma)[:-1]
		temporal_diff = buff.rewards + self.config.gamma * value_with_v[1:] - value_with_v[:-1]

		advantage = discount(temporal_diff, self.config.gamma)

		return disc_reward, advantage

	def sync(self):
		pass

	def update(self):
		pass