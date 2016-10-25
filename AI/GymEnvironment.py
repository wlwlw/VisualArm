import cv2
import gym
import random
import numpy as np

class ScreenEnvironment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)

        screen_width, screen_height = \
                config.screen_width, config.screen_height

        self.display = config.display
        self.dims = (screen_height, screen_width)

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        if self.lives == 0:
            self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @property
    def screen(self):
        return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_RGB2GRAY)/255., self.dims)
        #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def lives(self):
        return self.env.ale.lives()

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()

    def after_act(self, action):
        self.render()

class SimpleScreenEnvironment(ScreenEnvironment):
    def __init__(self, config):
        super(SimpleScreenEnvironment, self).__init__(config)

    def act(self, action):
        self._step(action)

        self.after_act(action)
        return self.state

class SimpleGymEnvironment(object):
    def __init__(self, config):
        self.env = gym.make(config.env_name)

        screen_width, screen_height = \
                config.screen_width, config.screen_height

        self.display = config.display
        self.dims = (screen_height,screen_width)

        self._screen = None
        self.reward = 0
        self.terminal = True

    def new_game(self, from_random_game=False):
        self._screen = self.env.reset()
        self._step(0)
        self.render()
        return self.screen, 0, 0, self.terminal

    def _step(self, action):
        self._screen, self.reward, self.terminal, _ = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @property
    def screen(self):
        return np.reshape(self._screen,self.dims)
        #return cv2.resize(cv2.cvtColor(self._screen, cv2.COLOR_BGR2YCR_CB)/255., self.dims)[:,:,0]

    @property
    def action_size(self):
        return self.env.action_space.n

    @property
    def state(self):
        return self.screen, self.reward, self.terminal

    def render(self):
        if self.display:
            self.env.render()

    def act(self, action):
        self._step(action)
        self.render()
        return self.state



