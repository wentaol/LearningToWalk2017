from multiprocessing import Pool, TimeoutError
import time, os, argparse
import numpy as np
from myAgent import get_agent_and_env
import datetime as dt

def run_test(model_fname, agent_env_args):
    worker_id = os.getpid()
    print("Worker %d: Starting test..." % (worker_id))

    total_reward = 0

    agent, env = get_agent_and_env(**agent_env_args)

    agent.load_weights(model_fname)
    agent.training = False
    agent.step = 0
    agent._on_test_begin()
    agent.reset_states()

    processor = agent.processor

    np.random.seed()
    seed = np.random.randint(2**32)
    print("Worker %d: Seed is %d" % (worker_id, seed))
    ma = agent.processor.mirror_action
    mo = agent.processor.mirror_observation
    observation = env.reset(seed=np.random.randint(2**32))
    observation = mo(observation)
    # observation, _, _, _ = processor.process_step(observation, 0, False, None)
    total_reward = 0.
    
    while True:
        action = agent.forward(observation)
        action = ma(action)
        observation, reward, done, info = env.step(action.tolist())
        observation = mo(observation)
        # observation, reward, done, info = processor.process_step(observation, reward, done, info)

        total_reward += reward

        if done:
            break

    print("Worker %d: Reward = %.3f" % (worker_id, total_reward))

    return total_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
    parser.add_argument('model')
    parser.add_argument('--pool-size', dest='pool_size', type=int, default=1)
    parser.add_argument('--run-count', dest='run_count', type=int, default=1)
    parser.add_argument('--visualize', action="store_true", default=False)
    # parser.add_argument('--timeout', type=int, help="Number of seconds before individual test run is killed.")
    args = parser.parse_args()

    pool = Pool(processes=args.pool_size)

    # Arguments for the agent and environment
    agent_env_args = dict(
        step_size=0.01, visualize=args.visualize, # OsimEnv
        print_summary=False, test_mode=True,
        )


    # launching multiple evaluations asynchronously *may* use more processes
    start_time = dt.datetime.now()
    test_results = [pool.apply_async(run_test, (args.model, agent_env_args)) for i in range(args.run_count)]
    rewards = [res.get() for res in test_results]
    mean, min_reward, max_reward = np.mean(rewards), min(rewards), max(rewards)
    print("Mean test reward over %d runs in %.1f seconds: %.3f [%.3f, %.3f]" % (args.run_count, (dt.datetime.now() - start_time).total_seconds(),mean, min_reward, max_reward))
