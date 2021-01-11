from sampler import Sampler
from myWorker import DDPGWorker
import argparse

# Command line parameters
parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--train', dest='train', action='store_true', default=False)
parser.add_argument('--test', dest='test', action='store_true', default=False)
parser.add_argument('--steps', dest='steps', action='store', default=10000, type=int)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--model', dest='model', action='store', default="example.h5f")
parser.add_argument('--model-load', dest='model_load', action='store', default="")
parser.add_argument('--memory-load', dest='memory_load', action='store', default="")
parser.add_argument('--submit', dest='submit', action='store_true', default=False, required=False)
parser.add_argument('--token', dest='token', action='store', required=False)
parser.add_argument('--n-workers', dest='n_workers', type=int, action='store', required=False, default=2)
parser.add_argument('--sampler-update-interval', dest='sampler_update_interval', type=int, action='store', required=False, default=1000)
parser.add_argument('--noenv', dest='noenv', action='store_true', default=False)
args = parser.parse_args()

# Arguments for the agent and environment
agent_env_args = dict(
    nb_steps_warmup=100, 
    memory_limit=200000, 
    ou_theta=0.3, ou_mu=0., ou_sigma=0.4, # OrnsteinUhlenbeckProcess
    gamma=0.98, target_model_update=1e-5, batch_size=128, # ParallelDDPGAgent
    actor_lr=0.0003, critic_lr=0.001, delta_clip=1.0,
    step_size=0.05, visualize=False, # OsimEnv
    print_summary=False,
    )

# See myAgent.py for configuring the agent and environment. However most of the arguments are here.

if args.train:
    sampler = Sampler(n_workers=args.n_workers)
    worker = DDPGWorker()
    worker_args = dict(agent_env_args=agent_env_args, env_reset_interval=10000000)
    sampler.start_workers(worker, worker_args)

agent_env_args['memory_path']=args.memory_load

import os
#os.environ['CUDA_VISIBLE_DEVICES']="0"
import sys
import math
import time
import pprint
import opensim as osim
import numpy as np
from rl.callbacks import FileLogger
from wentaoCallbacks import MyModelCheckpoint, MoreStatsCheckpoint
from osim.http.client import Client
from myAgent import get_agent_and_env
from multiprocessing import Pool

# Tweaking parameters
WINDOW_LENGTH=1         # I have no idea what this does. Wentao can figure out?
ACTION_REPETITION=1     # Each action is repeated this number of times. Useful when each action hardly affects the environment.
NB_STEPS_WARMUP=32     # How many steps to generate in the environment before engaging the net.
TRAIN_VERBOSE_LVL=2     # 0 for no logging, 1 for interval logging, 2 for episode logging
TRG_PER_EXP = 10

if args.train:
    agent_env_args['print_summary'] = True
    agent, env = get_agent_and_env(**agent_env_args)

    agent.training = True

    # Load weights from a file
    if len(args.model_load) > 0:
        agent.load_weights(args.model_load)

    # Send the weights of the master agent to the worker agents
    sampler.update_agent(agent)

    # Configure callbacks
    checkpoint_weights_filename = args.model + '_weights_{step}'
    log_filename = args.model + '_log.json'
    callbacks = [MyModelCheckpoint(checkpoint_weights_filename, interval=10000)]
    callbacks += [FileLogger(log_filename, interval=100)]
    callbacks += [MoreStatsCheckpoint("%s_morestats.csv" % args.model)]

    n_experiences_accumulated = 0

    # Initialize the parallel training
    agent.init_fit_parallel(callbacks=callbacks, nb_steps=args.steps, visualize=False,
                            verbose=TRAIN_VERBOSE_LVL, nb_max_episode_steps=None, log_interval=10000, 
                            action_repetition=ACTION_REPETITION)
    while True:
        if args.noenv:
            experience_list = []
            n_experiences_accumulated += args.sampler_update_interval
            n_backward_calls = args.sampler_update_interval
        else:
            experience_list = sampler.collect_data()
            if len(experience_list) > args.sampler_update_interval:
                experience_list = experience_list[-args.sampler_update_interval:]
            n_experiences_accumulated += len(experience_list)
            maxextra = args.sampler_update_interval * TRG_PER_EXP
            n_backward_calls = min(maxextra, len(experience_list) * TRG_PER_EXP)
        agent.fit_parallel(experience_list, 
                            sampler_update_interval=args.sampler_update_interval, 
                            n_backward_calls=n_backward_calls)
        if n_experiences_accumulated >= args.sampler_update_interval:
            n_experiences_accumulated = 0
            if not args.noenv:
                sampler.update_agent(agent)
    
    agent.save_weights(args.model, overwrite=True)

# TEST
if args.test:
    agent_env_args['visualize'] = args.visualize
    agent_env_args['test_mode'] = True
    agent, env = get_agent_and_env(**agent_env_args)

    agent.load_weights(args.model)
    agent.training = False
    agent.step = 0
    agent._on_test_begin()
    agent.reset_states()

    processor = agent.processor
    
    #p = Pool(8)
    #result = p.map(test, range(8))
    #result = [test(i) for i in range(1)]

    observation = env.reset()
    total_reward = 0.
    
    while True:
        action = agent.forward(observation)
        observation, reward, done, info = env.step(action.tolist())
        # observation, reward, done, info = processor.process_step(observation, reward, done, info)

        total_reward += reward

        if done:
            break
    print(total_reward)
    #print("=======================================")
    #print("Mean: {0} Min: {1} Max: {2}".format(
    #        np.mean(result), np.min(result), np.max(result)))
    #print("=======================================")
    

    # agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=None)


# SUBMIT
if args.submit:
    agent_env_args['visualize'] = False
    agent_env_args['test_mode'] = True
    pp = pprint.PrettyPrinter(indent=4)
    remote_base = 'http://grader.crowdai.org:1729'

    # Prepare the remote environment. NOTE: visualize = True doesn't work for remote env
    client = Client(remote_base)
    observation = client.env_create(args.token)

    # Prepare the agent
    agent, env = get_agent_and_env(**agent_env_args)
    agent.load_weights(args.model)
    agent.training = False
    agent.step = 0
    agent._on_test_begin()
    agent.reset_states()
    processor = agent.processor
    observation, reward, done, info = processor.process_step(observation, 0.0, False, dict())

    # Keeping count
    current_ep = 1
    total_reward = 0.
    past_rewards = []
    
    while True:
        action = agent.forward(observation)
        [observation, reward, done, info] = client.env_step(action.tolist())
        observation, newreward, done, info = processor.process_step(observation, reward, done, info)

        total_reward += reward
        past_rewards_str = ", ".join(["[%d] %.2f|%d" % (i + 1, r, s) for i, (r, s) in enumerate(past_rewards)])
        print("CURRENT REWARD (Ep %d, Step %d) ==> %.2f | [EP] REWARD|STEPS ==> %s" % (
            current_ep, agent.step, total_reward, past_rewards_str))
        # agent.backward(reward, terminal=done)
        agent.step += 1
        if done:
            observation = client.env_reset()

            # Keeping count
            past_rewards += [(total_reward, agent.step)]
            total_reward = 0.
            current_ep += 1

            # Reset the agent
            agent.step = 0
            agent.reset_states()

            if not observation:
                break
    client.submit()
