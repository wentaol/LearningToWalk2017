import numpy as np
import os
import sys
from heapq import *
import time
from collections import namedtuple

from base_model import Experience, ExperienceWithMoreStats
from base_worker import BaseWorker

from myAgent import get_agent_and_env


class DDPGWorker(BaseWorker):
    def run(self, agent_env_args=dict(), env_reset_interval=10000):
        print(os.getpid(), "Starting up worker_run().")

        # set any environmental variables
        #os.environ['OMP_NUM_THREADS'] = 1
        os.environ['KERAS_BACKEND']="tensorflow"
        os.environ['CUDA_VISIBLE_DEVICES']=""

        # Initialize agent and environment
        print(env_reset_interval)
        agent, env = get_agent_and_env(**agent_env_args)
        np.random.seed()
        curseed = np.random.randint(2**32-1)
        prev_observation = env.reset(seed=curseed)
        agent.training = True

        # initialize local worker variables
        weights_loaded = False
        total_steps = 0
        time.sleep(np.random.randint(0,30))
        totalreward = 0.0
        episode_number = 0
        workerid = os.getpid()
        # main process loop
        while True:
            while self.has_pending_action():
                action, args = self.get_pending_action()
                if action == 'update_agent':
                    weights = args['weights']
                    agent.set_weights(weights)
                    weights_loaded = True
                    agent.reset_state()

            if not weights_loaded:
                print(os.getpid(), "Weights not loaded, sleeping.")
                time.sleep(0.5)
                continue

            action = agent.processor.process_action(agent.get_action(prev_observation))
            new_observation, reward, done, info = env.step(action)
            agent.post_step(new_observation, reward, done, info)
            totalreward += reward

            e = ExperienceWithMoreStats(
                state0=np.array([prev_observation]),
                action=action,
                reward=reward,
                state1=np.array([new_observation]),
                terminal1=done,
                workerid=workerid,
                epnum=episode_number,
                cumulativereward=totalreward,
                seed=curseed
                )
            self.enq_out_data(e)

#            ma = agent.processor.mirror_action
#            mo = agent.processor.mirror_observation
#            e2 = ExperienceWithMoreStats(
#                state0=np.array([mo(prev_observation)]),
#                action=ma(action),
#                reward=reward,
#                state1=np.array([mo(new_observation)]),
#                terminal1=done,
#                workerid=workerid,
#                epnum=episode_number,
#                cumulativereward=totalreward,
#                seed=curseed
#                )
#            self.enq_out_data(e2)

            prev_observation = new_observation
            if done:
                # Update our outcomes
                #elem = (totalreward, curseed)
                #if len(listofshame) >= maxlen and elem < max(listofshame) :
                #    listofshame.remove(max(listofshame))
                #listofshame.append(elem)
                # Reset
                #print "Seed: %d Total reward: %.2f" % (curseed, totalreward)
                curseed = np.random.randint(2**32-1)
                prev_observation = env.reset(seed=curseed)
                agent.reset_state()
                totalreward = 0.0
                episode_number += 1

            total_steps += 1
            #if total_steps % env_reset_interval == 0:
                #env = get_env("train")
                #prev_observation = env.reset()
                #agent.reset_state()



