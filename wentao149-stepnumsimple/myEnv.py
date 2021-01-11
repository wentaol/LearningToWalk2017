import numpy as np
from osim.env import RunEnv
from myProcessor import StateProcessor

# Sampler libraries
from base_env import Env

# def get_env(env_type="train", visualize=False):
#     if env_type == "train":
#         stepsize = 0.05
#         timestep_limit = 1000*0.01/stepsize
#         processor = StateProcessor(stepsize=stepsize)
#         return OsimEnv(visualize=visualize, processor=processor, stepsize=stepsize, timestep_limit=timestep_limit)
#     elif env_type == "test":
#         stepsize = 0.05
#         timestep_limit = 1000*0.01/stepsize
#         processor = StateProcessor()
#         return OsimEnv(visualize=visualize, processor=processor, stepsize=stepsize, timestep_limit=timestep_limit)

class OsimEnv(Env):
    def __init__(self, visualize=True, test=False, step_size=0.01, processor=None, timestep_limit=1000):
        self.visualize = visualize
        self._osim_env = RunEnv(visualize=visualize)
        self._osim_env.stepsize = step_size
        self._osim_env.spec.timestep_limit = timestep_limit
        self._osim_env.horizon = timestep_limit
        # self._osim_env.integration_accuracy = 1e-1
        if test:
            self._osim_env.timestep_limit = 1000
        self.processor = processor
        print "stepsize: " + str(self._osim_env.stepsize)

    def reset(self,seed=None,difficulty=2):
        observation = self._osim_env.reset(seed=seed,difficulty=difficulty)
        if self.processor:
            observation, reward, done, info = self.processor.process_step(observation, 0.0, False, dict())

        return observation

    def step(self, action):
        if self.processor:
            action = self.processor.process_action(action)
        
        observation, reward, done, info = self._osim_env.step(action)
        
        if self.processor:
            observation, reward, done, info = self.processor.process_step(observation, reward, done, info)
        
        return observation, reward, done, info

    def get_observation_dim(self):
        return len(self.reset())

    def get_action_dim(self):
        nb_actions = self._osim_env.action_space.shape[0]
        return nb_actions

    # FOR PICKLING
    def __setstate__(self, state):
        self.__init__(visualize=state['visualize'])
    def __getstate__(self):
        state = {'visualize': self.visualize}
        return state
