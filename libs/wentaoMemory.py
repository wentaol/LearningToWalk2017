from __future__ import absolute_import
from collections import deque, namedtuple
import warnings
import random
import cPickle as pickle
import gzip
import numpy as np


# This is to be understood as a transition: Given `state0`, performing `action`
# yields `reward` and results in `state1`, which might be `terminal`.
Experience = namedtuple('Experience', 'state0, action, reward, state1, terminal1')

class RingBuffer(object):
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = None
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):                
        if idx < 0 or idx >= self.length:
            raise KeyError()
        ret = self.data[(self.start + idx) % self.maxlen, :]        
        return ret

    def getSamples(self, idxList):
        idxList2 = [(self.start + idx) % self.maxlen for idx in idxList]
        return self.data[idxList2, :]

    def append(self, v):        
        if self.length < self.maxlen:
            # We have space, simply increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, "remove" the first item.
            self.start = (self.start + 1) % self.maxlen
        else:
            # This should never happen.            
            raise RuntimeError()
        if type(self.data) != np.ndarray:
            if(type(v) == np.ndarray):
                self.data = np.zeros((self.maxlen, v.size))
            elif(type(v) == list):
                self.data = np.zeros((self.maxlen, len(v)))
            else:
                self.data = np.zeros((self.maxlen, 1))
        self.data[(self.start + self.length - 1) % self.maxlen, :] = v

class WeaklyOrderedMemory():
    def __init__(self, limit,  **kwargs):
        self.limit = limit
        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        # Directly store the episodes for simplicity at the cost of 2 times the memory
        self.state0 = RingBuffer(limit)
        self.action = RingBuffer(limit)
        self.reward = RingBuffer(limit)
        self.state1 = RingBuffer(limit)
        self.terminal1 = RingBuffer(limit)

    def __len__(self):
        return len(self.state0)
    
    def _get_random_idxs(self, length, size=32):
        return np.random.randint(0, length, size=size)
 
    def sample(self, batch_size):
        if batch_size > len(self.state0):
            batch_idxs = np.random.choice(len(self.state0), batch_size, 
                                    replace=batch_size > len(self.state0))
        else:
            batch_idxs = random.sample(xrange(len(self.state0)), batch_size)
        state0 = self.state0.getSamples(batch_idxs)
        action = self.action.getSamples(batch_idxs)
        reward = self.reward.getSamples(batch_idxs)
        state1 = self.state1.getSamples(batch_idxs)
        terminal1 = self.terminal1.getSamples(batch_idxs)
        return state0, action, reward, state1, terminal1

    def append(self, episode):
        self.state0.append(episode.state0)
        self.action.append(episode.action)
        self.reward.append(episode.reward)
        self.state1.append(episode.state1)
        tmp = 0. if episode.terminal1 else 1.
        self.terminal1.append(tmp)

    def save(self, path):
        f = gzip.open(path, 'wb')
        pickle.dump(self, f)
        f.close()
    
    def load(self, path):
        f = gzip.open(path, 'rb')
        tmp = pickle.load(f)
        self.limit = tmp.limit
        # Do not use deque to implement the memory. This data structure may seem convenient but
        # it is way too slow on random access. Instead, we use our own ring buffer implementation.
        # Directly store the episodes for simplicity at the cost of 2 times the memory
        self.state0 = tmp.state0
        self.action = tmp.action
        self.reward = tmp.reward
        self.state1 = tmp.state1
        self.terminal1 = tmp.terminal1
        f.close()
        

