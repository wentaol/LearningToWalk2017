from rl.core import Processor
import numpy as np
import math
from collections import deque
import itertools

DEBUG           = False # Turns on or off debug messages
STEPTHRU        = False

REWARD_MULT = 50

def DPRINT(s):
    if DEBUG:
        print(s)

class StateProcessor(Processor):
    def __init__(self, step_size=0.01, test_mode=False):
        super(StateProcessor, self).__init__()
        self._reset()
        self.step_size = step_size
        self.testmode = test_mode

    def _reset(self):
        self.numsteps = 0;
        #self.prev_obs = None
        #self.talus_pos = deque()  
        return

    def process_step(self, observation, reward, done, info):
        pelvis_pos_x = observation[1]
        pelvis_vel_x = observation[4]
        cent_mass_pos_x = observation[18]
        cent_mass_vel_x = observation[20]
        pelvis_x = observation[24]
        pelvis_y = observation[25]
        head_x = observation[22]
        # print(pelvis_pos_x, pelvis_vel_x, cent_mass_pos_x, cent_mass_vel_x, pelvis_y)
        # Reward multiplier
        #DPRINT("BASE    REWARD\t= %.6f" % reward)
        #horizontal_vel_reward = cent_mass_vel_x * 0.5 * 0.001   # Horizontal velocity always increasing
        #DPRINT("HORIVEL REWARD\t= %.6f" % horizontal_vel_reward)
        #vert_pos_reward = np.clip(((pelvis_y - 0.85)) * 0.2, -1, 0)
        #DPRINT("VERTPOS REWARD\t= %.6f" % vert_pos_reward)
        #DPRINT("HDSYNC PENLTY\t= %.6f" % head_hip_sync_penalty)
        #reward += vert_pos_reward 
        # Reward clipping
        reward = self.process_reward(reward)
        #DPRINT("> FINAL REWARD\t= %.6f" % reward)
        #if STEPTHRU:
        #    raw_input("Press ENTER to continue")
        # self.last_vel_x = cent_mass_vel_x
        # Aggressive early done conditions
        #hrelx = head_x - pelvis_x
        #if hrelx < -0.4 or 0.6 < hrelx:  
        #    done = True
        newobs = self.process_observation(observation)
        #self.numsteps += 1
        return newobs, reward, done, info
    # These is this mother function:
    def process_reward(self, reward):
        if not self.testmode:
            reward *= REWARD_MULT
        return reward


    def process_observation(self, observation):
        """Processes the observation as obtained from the environment for use in an agent and
        returns it.
        """
        #ptheta = observation[0]
        #ppos = np.array(observation[1:3])
        #parts = np.array(observation[22:36]).reshape((7,2))
        #rotate body parts into pelvis frame
        #relpos = parts - ppos
        #c, s = np.cos(ptheta), np.sin(ptheta)
        #rot = np.array([[c, -s],[s, c]])
        #newparts = relpos.dot(rot)
        #newobs = np.concatenate((observation, newparts.flatten()))
        # set obstacle position relative to pelvisi
        newobs = observation[:]
        newobs[18] -= observation[1] # cm X
        newobs[22] -= observation[1] # head X
        newobs[24] = 0.0 # pelvis_x (dup)
        newobs[25] = self.numsteps * self.step_size # pelvis_y (dup)
        newobs[26] -= observation[1] # torso x
        newobs[28] -= observation[1] # toesl x
        newobs[30] -= observation[1] # toesr x
        newobs[32] -= observation[1] # talusl x
        newobs[34] -= observation[1] # talusr x
        newobs[1] = 0.0        
        DPRINT("Obs: " + str(newobs))
        return newobs

    def process_info(self, info):
        """Processes the info as obtained from the environment for use in an agent and
        returns it.
        """
        return info

    def process_action(self, action):
        """Processes an action predicted by an agent but before execution in an environment.
        """
	#print(s)
	return np.clip(action, 0., 1.)
        # return action

    def process_state_batch(self, batch):
        """Processes an entire batch of states and returns it.
        """
        return batch
        # EXAMPLE from https://github.com/matthiasplappert/keras-rl/blob/master/examples/dqn_atari.py
        # We could perform this processing step in `process_observation`. In this case, however,
        # we would need to store a `float32` array instead, which is 4x more memory intensive than
        # an `uint8` array. This matters if we store 1M observations.
        # processed_batch = batch.astype('float32') / 255.
        # return processed_batch
from rl.core import Processor
import numpy as np
import math
