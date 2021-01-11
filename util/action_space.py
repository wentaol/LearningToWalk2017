# Plots the action sample space to see the range

from osim.env import RunEnv
import matplotlib.pyplot as plt

env = RunEnv(visualize=False)
env.reset(difficulty=0)
samples = [env.action_space.sample() for i in range(300)]
y1 = [v[0] for v in samples]
plt.plot(samples)
plt.show()
