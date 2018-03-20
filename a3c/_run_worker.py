import argparse
import gym
import tensorflow as tf
import time

from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.cmd_util import make_atari_env
from _a3c import Worker
from tensorflow.python.client import device_lib


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
parser.add_argument('--max_rollouts', type=int, default=5,
                    help='max rollouts')
parser.add_argument('--task_index', type=int, default=0,
                    help='task index')
parser.add_argument('--job_name', type=str, default='worker',
                    help='job name')

def run(config, server):
	seed = 123
	tf.set_random_seed(seed)

	env = VecFrameStack(make_atari_env(config.env, config.num_env, seed), 4)

	config.input_dims = [84, 84, 4]
	config.output_dims = env.action_space.n
	config.scope = 'worker'
	config.reuse = tf.AUTO_REUSE

	sess = tf.Session()
	worker = Worker(env, 'worker', sess, 'results/', config)

	#sv = tf.train.Supervisor(logdir='outputs', is_chief=(config.task_index == 0))

	#print(device_lib.list_local_devices())
	print(server.target)

	with tf.Session(target=server.target, config=tf.ConfigProto(allow_soft_placement=True)) as sess:#, sess.as_default():
		sess.run(tf.global_variables_initializer())
		sess.run(tf.local_variables_initializer())
		print(worker._sync.device)
		worker.sess = sess
		worker.actor.sess = sess
		worker.global_actor.sess = sess
		while worker.total_iterations < 10e6:
			worker.run()

def main(_):
	config = parser.parse_args()

	cluster = tf.train.ClusterSpec({"ps": ["localhost:8000"], "worker": ["localhost:8001", "localhost:8002", "localhost:8003"]})
	print(cluster.as_cluster_def())

	if config.job_name == "worker":
		server = tf.train.Server(cluster, job_name="worker", task_index=config.task_index,
			config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=2))
		run(config, server)
	else:
		server = tf.train.Server(cluster, job_name="ps", task_index=config.task_index,
			config=tf.ConfigProto(device_filters=["/job:ps"]))
		while True:
			time.sleep(1000)

if __name__ == "__main__":
    tf.app.run()