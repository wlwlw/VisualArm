import random
import tensorflow as tf
from AI.agent import Agent
from AI.ROSEnvironment import ArmControlPlateform
from config import get_config

flags = tf.app.flags

# Model
flags.DEFINE_string('model', 'DQN', 'Type of model')
# Environment
#flags.DEFINE_string('env_name', 'Acrobot-v1', 'The name of gym environment to use')
# Etc
#flags.DEFINE_boolean('display', True, 'Whether to do display the game screen or not')
flags.DEFINE_integer('random_seed', 123, 'Value of random seed')

FLAGS = flags.FLAGS

# Set random seed
tf.set_random_seed(FLAGS.random_seed)
random.seed(FLAGS.random_seed)

def main(_):
    config = get_config(FLAGS) or FLAGS
    env = ArmControlPlateform(config)
    for _ in range(1):
        env.reset()
        agent = Agent(config, env)
        agent.activate(learn=True)

def test():
    config = get_config(FLAGS) or FLAGS
    env = ArmControlPlateform(config)
    env.reset()
    for i in range(30):
        sleep(0.1)
        action = env.action_space.sample()
        ob,re,ter = env.step(action)
        print(ob,re,ter)

if __name__ == '__main__':
    tf.app.run()
    #test()