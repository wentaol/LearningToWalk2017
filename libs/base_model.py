from collections import namedtuple

Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')
ExperienceWithMoreStats = namedtuple('ExperienceWithMoreStats', 'state0, action, reward, state1, terminal1, workerid, epnum, cumulativereward, seed')

class Model(object):
    def get_action(self, observation):
        raise NotImplementedError

    def post_step(self, observation, reward, done, info):
        raise NotImplementedError

    def set_weights(self, weights):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def reset_state(self):
        raise NotImplementedError