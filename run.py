import argparse
import gym
import tensorflow as tf

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env
from a3c import Worker
# from pytorch_a3c import Worker

parser = argparse.ArgumentParser(description='A3C')

parser.add_argument('--env', default = 'PongNoFrameskip-v4',
                    help='environment to test on')
parser.add_argument('--num_env', type=int, default = '1',
                    help='number of envs')
parser.add_argument('--vf_coeff', type=float, default=0.5,
                    help='value function loss coefficient')
parser.add_argument('--entropy_coeff', type=float, default=0.01,
                    help='entropy loss coefficient')
parser.add_argument('--lr', type=float, default=7e-4,
                    help='learning rate')
parser.add_argument('--max_grad_norm', type=float, default=0.5,
                    help='maximum gradient norm for clipping')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor')
parser.add_argument('--history', type=int, default=4,
                    help='number of frames in the past to keep in the state')
parser.add_argument('--policy_type', default='cnn',
                    help='policy architecture')
parser.add_argument('--reuse', type=bool, default=False,
                    help='policy architecture')
parser.add_argument('--batch_size', type=int, default=64,
                    help='batch size')
parser.add_argument('--normalize_adv', type=bool, default=True,
                    help='normalize advantage')

if __name__ == '__main__':
	config = parser.parse_args()

	seed = 123
	tf.set_random_seed(seed)

	env = VecFrameStack(make_atari_env(config.env, config.num_env, seed), 4)
	sess = tf.Session()

	config.input_dims = [84, 84, 4]
	config.output_dims = env.action_space.n
	config.scope = 'worker'
	config.reuse = tf.AUTO_REUSE

	worker = Worker(env, 'worker', sess, 'results/', config)

	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())

	with sess.as_default():
		while worker.total_iterations < 10e6:	
			reward = worker.run()

# if __name__ == '__main__':
# 	config = parser.parse_args()

# 	env = gym.make(config.env)
# 	#sess = tf.Session()
# 	sess = 'dummy'

# 	config.input_dims = [80, 80, 4]
# 	config.output_dims = env.action_space.n
# 	config.scope = 'worker'

# 	worker = Worker(env, 'worker', sess, 'results-pytorch/', config)

# 	#sess.run(tf.global_variables_initializer())
# 	#sess.run(tf.local_variables_initializer())

# 	while worker.total_iterations < 5e6:
# 		#with sess.as_default():
# 		worker.run()