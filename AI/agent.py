__author__ = 'Bright'
import tensorflow as tf
import numpy as np
import random
from functools import reduce
from lib.base import BaseModel
from lib.ops import conv2d, linear
from lib.history import History
from lib.replay_memory import ReplayMemory
from time import sleep

def build_DQN(s_t, action_size, target_q_t, action, learning_rate_step,cnn_format = 'NHWC'):

    min_delta = -1
    max_delta = 1
    learning_rate_initial = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 50

    w = {}
    #initializer = tf.contrib.layers.xavier_initializer()
    initializer = tf.truncated_normal_initializer(0, 0.02)
    activation_fn = tf.nn.relu

    with tf.variable_scope('Q_network'):

        l1, w['l1_w'], w['l1_b'] = conv2d(s_t,
            32, [8, 8], [4, 4], initializer, activation_fn, cnn_format, name='l1')
        l2, w['l2_w'], w['l2_b'] = conv2d(l1,
            64, [4, 4], [2, 2], initializer, activation_fn, cnn_format, name='l2')
        l3, w['l3_w'], w['l3_b'] = conv2d(l2,
            64, [3, 3], [1, 1], initializer, activation_fn, cnn_format, name='l3')

        shape = l3.get_shape().as_list()
        l3_flat = tf.reshape(l3, [-1, reduce(lambda x, y: x * y, shape[1:])])

        l4, w['l4_w'], w['l4_b'] = linear(l3_flat, 512, activation_fn=activation_fn, name='l4')
        q, w['q_w'], w['q_b'] = linear(l4, action_size, name='q')

        q_summary = []
        avg_q = tf.reduce_mean(q, 0)
        for idx in range(action_size):
            q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
        q_summary = tf.merge_summary(q_summary, 'q_summary')

    with tf.variable_scope('optimzier'):

        action_one_hot = tf.one_hot(action, action_size, 1.0, 0.0, name='action_one_hot')
        q_acted = tf.reduce_sum(q * action_one_hot, reduction_indices=1, name='q_acted')

        delta = target_q_t - q_acted
        clipped_delta = tf.clip_by_value(delta, min_delta, max_delta, name='clipped_delta')

        loss = tf.reduce_mean(tf.square(clipped_delta), name='loss')
        learning_rate = tf.maximum(learning_rate_minimum,
            tf.train.exponential_decay(
                learning_rate_initial,
                learning_rate_step,
                learning_rate_decay_step,
                learning_rate_decay,
                staircase=True))

        optim = tf.train.RMSPropOptimizer(
            learning_rate, momentum=0.95, epsilon=0.01).minimize(loss)

    return w, q, q_summary, optim, loss

def build_simpleQN(s_t, action_size, target_q_t, action, learning_rate_step):
    min_delta = -1
    max_delta = 1
    learning_rate_initial = 0.0025
    learning_rate_minimum = 0.0025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 100
    w = {}
    activation_fn = tf.nn.relu
    with tf.variable_scope('Q_network'):
        shape = s_t.get_shape().as_list()
        s_t_flat = tf.reshape(s_t, [-1, reduce(lambda x, y: x * y, shape[1:])])
        #l, w['l_w'], w['l_b'] = linear(s_t_flat, action_size+s_t_flat.get_shape().as_list()[-1], activation_fn=activation_fn, name='l')
        q, w['q_w'], w['q_b'] = linear(s_t_flat, action_size, name='q')
        q_summary = []
        avg_q = tf.reduce_mean(q, 0)
        for idx in range(action_size):
            q_summary.append(tf.histogram_summary('q/%s' % idx, avg_q[idx]))
        q_summary = tf.merge_summary(q_summary, 'q_summary')
    with tf.variable_scope('optimzier'):
        action_one_hot = tf.one_hot(action, action_size, 1.0, 0.0, name='action_one_hot')
        q_acted = tf.reduce_sum(q * action_one_hot, reduction_indices=1, name='q_acted')
        delta = target_q_t - q_acted
        clipped_delta = tf.clip_by_value(delta, min_delta, max_delta, name='clipped_delta')
        loss = tf.reduce_mean(tf.square(clipped_delta), name='loss')
        learning_rate = tf.maximum(learning_rate_minimum,
          tf.train.exponential_decay(
              learning_rate_initial,
              learning_rate_step,
              learning_rate_decay_step,
              learning_rate_decay,
              staircase=True))
        #optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        optim = tf.train.RMSPropOptimizer(learning_rate, momentum=0.95, epsilon=0.01).minimize(loss)
    return w, q, q_summary, optim, loss

class Agent(BaseModel):
    def __init__(self, config, environment, model_file_path=None):
        super(Agent, self).__init__(config)
        self.model_file_path = model_file_path
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        #self.weight_dir = 'weights'
        self.env = environment
        self.history = History(self.config)
        self.memory = ReplayMemory(self.config, self.model_dir)
        self.step = 0
        with self.graph.as_default():
            with tf.variable_scope(self.name):
                self.build_connectome()

    def decide(self, s_t, test_ep=None):
        ep = (
            self.ep_end+max(0.,(self.ep_start-self.ep_end)*(self.ep_end_t-max(0.,self.step-self.learn_start_step))/self.ep_end_t)
            ) if test_ep is None else test_ep

        if random.random() < ep:
            action = random.randrange(self.env.action_space.n)
        else:
            action = self.sess.run(self.q_action, {self.s_t: [s_t]})[0]
        #print(action)
        return action

    def observe(self, screen, reward, action, done):
        reward = max(self.min_reward, min(self.max_reward, reward))
        self.memory.add(screen, reward, action, done)

        if self.step > self.learn_start_step and self.step % self.train_frequency == 0:
            if self.memory.count < self.history_length:
                return
            else:
                s_t, action, reward, s_t_plus_1, done = self.memory.sample()
                s_t, s_t_plus_1 = np.transpose(s_t,(0,2,1) if len(self.observation_space)==1 else (0,2,3,1)),np.transpose(s_t_plus_1,(0,2,1) if len(self.observation_space)==1 else (0,2,3,1))

            done = np.array(done) + 0.
            q_t_plus_1 = self.sess.run(self.q, {self.s_t: s_t_plus_1})
            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)
            target_q_t = (1.-done)*self.discount * max_q_t_plus_1 + reward

            _, q_t, loss, summary_str = self.sess.run([self.optim, self.q, self.loss, self.q_summary], {
                self.target_q_t: target_q_t,
                self.action: action,
                self.s_t: s_t,
                self.learning_rate_step: self.step
            })

            #self.writer.add_summary(summary_str, self.step)
            self.total_loss += loss
            self.total_q += q_t.mean()

    def activate(self, learn = True):
        total_reward = 0.
        self.total_loss, self.total_q = 0., 0.
        observation = self.env.observation
        for _ in range(self.history_length):
            self.history.add(observation)

        for self.step in range(self.max_step):
            # 1. decide
            action = self.decide(np.transpose(
                    self.history.get(), 
                    (1, 2, 0) if len(self.observation_space)==2 else (1,0)
                ), test_ep=None if learn else 0.
            )
            # 2. act
            observation, reward, done = self.env.step(action)
            # 3. observe
            total_reward += reward
            self.history.add(observation)
            if learn is True:
                self.observe(observation, reward, action, done)
            if done: 
                #break
                print('step: %d, total_r: %.4f' % (self.step, total_reward))
                self.env.reset()
                total_reward = 0.

            if self.step >= self.learn_start_step and learn is True:
                if self.step % self.test_step == self.test_step - 1:
                    avg_loss = self.total_loss / (self.test_step/self.train_frequency)
                    avg_q = self.total_q / (self.test_step/self.train_frequency)
                    self.total_loss, self.total_q = 0., 0.

                    print('step: %d, avg_l: %.6f, avg_q: %3.6f' % (self.step, avg_loss, avg_q))
                    tag_dict = dict(zip(self.summary_tags, [total_reward, avg_loss, avg_q]))
                    summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
                        self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
                    })
                    for summary_str in summary_str_lists:
                        self.writer.add_summary(summary_str, self.step)

            if self.step % self.save_step == self.save_step -1 and learn is True:
                self.save_model(filename=self.model_file_path, step=self.step)
    
                self.env.reset()
                total_reward = 0.
        self.sess.close()


    def build_connectome(self):
        # NHWC cnn_format
        self.s_t = tf.placeholder('float32',
            [None]+ self.observation_space+[self.history_length], name='s_t')
        if len(self.observation_space)==1:
            #print(self.s_t.get_shape())
            #shape = self.s_t.get_shape().as_list()
            self._s_t = tf.reshape(self.s_t,[-1,self.observation_space[0],1,self.history_length])
        elif len(self.observation_space)==2:
            self._s_t = self.s_t
        else:
            raise AssertionError("Only 1 or 2 dimension observation_space is supported.")


        self.target_q_t = tf.placeholder('float32', [None], name='target_q_t')
        self.action = tf.placeholder('int64', [None], name='action')
        self.learning_rate_step = tf.placeholder('int64', None, name='learning_rate_step')

        self.w, self.q, self.q_summary, self.optim, self.loss = \
            build_simpleQN(self._s_t, self.env.action_space.n, self.target_q_t, self.action, self.learning_rate_step)

        self.q_action = tf.argmax(self.q, dimension=1)

        with tf.variable_scope('summary'):
            self.summary_tags = ['total_reward', 'average_loss', 'average_q']

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in self.summary_tags:
                self.summary_placeholders[tag] = tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag]  = tf.scalar_summary("%s/%s" % (self.env_name, tag), self.summary_placeholders[tag])

            self.writer = tf.train.SummaryWriter('./logs/%s' % self.name, self.sess.graph)

        tf.initialize_all_variables().run(session=self.sess)
        self.saver = tf.train.Saver(max_to_keep=10)
        self.load_model(filename=self.model_file_path)