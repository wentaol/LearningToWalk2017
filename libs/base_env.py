import time
import numpy as np

class Env(object):
    def step(self, action):
        """ Returns (observation, reward, done, info)
        """
        raise NotImplementedError

    def reset(self):
        """ Returns the observation of the reset environment.
        """
        raise NotImplementedError

    def get_observation_dim(self):
        raise NotImplementedError

    def get_action_dim(self):
        raise NotImplementedError

class FakeEnv(Env):
    def __init__(self):
        self.step_count = 0

    def step(self, action):
        # print(os.getpid(), "FakeEnv.step() called.")

        time.sleep(0.1) # simulate computation

        self.step_count += 1
        observation = np.random.uniform(size=(self.get_observation_dim(),))
        reward = np.random.uniform()
        if self.step_count == 100:
            done = True
            self.step_count = 0
        else:
            done = False
        info = None
        return (observation, reward, done, info)

    def reset(self):
        self.step_count = 0

    def get_observation_dim(self):
        return 41

    def get_action_dim(self):
        return 18