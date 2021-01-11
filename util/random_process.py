# Shows what changing the parameters of the exploration process does

from rl.random import OrnsteinUhlenbeckProcess
import matplotlib.pyplot as plt

random_process = OrnsteinUhlenbeckProcess(theta=.3, mu=0., sigma=0.4, size=18)
# random_process = OrnsteinUhlenbeckProcess(theta=0.3, mu=0., sigma=0.1, size=18)
random_values = [random_process.sample() for i in range(100)]
# y1 = [x[0] for x in random_values]
plt.plot(random_values)
plt.show()
