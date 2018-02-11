from practicea3c import Worker, Actor, CNNPolicy, GAEOptimizer

import gym
import tensorflow as tf

env = gym.make('Pong-v0')

policyParams = {
	'scope': 'worker',
	'reuse': False#tf.AUTO_REUSE,
	'inputDims': [80, 80, 4],
	'outputDims': env.action_space.n
}

worker = Worker(env, 'worker', CNNPolicy, policyParams, 0.99, GAEOptimizer, 'results/')

while True:
	if worker.totalTransitions < 5e6:
		worker.runPolicy()

