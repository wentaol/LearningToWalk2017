from osim.env import RunEnv
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Train or test neural net motor controller')
parser.add_argument('--seed', type=int, default=None)
args = parser.parse_args()

env = RunEnv(visualize=True)
if not args.seed:
	seed = np.random.randint(2**32-1)
else:
	seed = args.seed
print("Seed = %d" % seed)
observation = env.reset(difficulty=2, seed=args.seed)
observation, reward, done, info = env.step(env.action_space.sample())



raw_input()