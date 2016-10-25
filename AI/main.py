__author__ = 'Bright'

import random

import tensorflow as tf

from GymEnvironment import SimpleGymEnvironment
from agent import Agent
from config import get_config

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'DQN', 'Type of model')
# Environment
#flags.DEFINE_string('env_name', 'Acrobot-v1', 'The name of gym environment to use')
# Etc
flags.DEFINE_boolean('display', True, 'Whether to do display the game screen or not')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

def main(_):
    config = get_config(FLAGS) or FLAGS
    env = SimpleGymEnvironment(config)
    for _ in range(1):
        env.new_game()
        agent = Agent(config, env)
        agent.activate(learn=True)

if __name__ == '__main__':
    tf.app.run()
