import numpy as np

class History:
    """
    history compact with cnn_format NCHW (number of sample in the batch, channel, height, width)
    """
    def __init__(self, config):
        batch_size, history_length, observation_space = \
                config.batch_size, config.history_length, config.observation_space

        self.history = np.zeros(
                [history_length]+list(observation_space), dtype=np.float32)

    def add(self, observation):
        self.history[:-1] = self.history[1:]
        self.history[-1] = observation

    def reset(self):
        self.history *= 0

    def get(self):
        return self.history
