#!/usr/bin/python

import argparse
import json
import numpy as np

def remove_outliers(l, threshold):
    prev, result = l[0], []
    for v in l:
        if np.abs(v - prev) > threshold:
            result.append(prev)
        else:
            result.append(v)
            prev = v
    return result

parser = argparse.ArgumentParser(description='Visualize keras-rl log files')
parser.add_argument('log_file')
parser.add_argument('--hide', dest='hide', action='store_true', default=False, help="Doesn't display the window with the graph")
parser.add_argument('--export', required=False, help="The path to export the PNG to.")
parser.add_argument('--morestats', required=False, type=str, default="", help="The path to the more stats file.")
args = parser.parse_args()

import matplotlib
if args.hide:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

def gethhmm(durations):
    secs = sum([float(x) for x in durations])
    return int(secs / 3600), int(int(secs) % 3600 / 60)

with open(args.log_file, 'r') as f:
    log_contents = f.read()

log = json.loads(log_contents)

episode = np.nan_to_num(log['episode'])
episode_reward = remove_outliers(np.nan_to_num(log['episode_reward']), 20000)
mean_q = remove_outliers(np.nan_to_num(log['mean_q']), 500)
episode_durations = log['duration']
hh, mm = gethhmm(episode_durations)
nb_episode_steps = log['nb_episode_steps']
nb_steps = log['nb_steps']
total_steps = int(nb_steps[-1])

print("Total steps: {:,}".format(total_steps))
print("Total run time: %d hrs %d mins" % (hh, mm))

fig = plt.figure(1, figsize=(20, 9), dpi=90)

plt.subplot(221)
plt.title('main thread episode_reward')
plt.xlabel('Episode')
x, y = np.array(episode), np.array(episode_reward)
fit = np.polyfit(x, y, deg=1)
f = np.poly1d(fit)
y_fit = f(x)
plt.scatter(x, y, s=0.5)
plt.plot(x, y_fit, color='red')

plt.subplot(222)
plt.title('mean_q')
plt.xlabel('Episode')
x, y = np.array(episode), np.array(mean_q)
fit = np.polyfit(x, y, deg=1)
f = np.poly1d(fit)
y_fit = f(x)
plt.plot(x, y)
plt.plot(x, y_fit, color='red')

# plt.subplot(233)
# plt.title('steps/ep')
# plt.xlabel('Episode')
# x, y = np.array(episode), np.array(nb_episode_steps)
# fit = np.polyfit(x, y, deg=1)
# f = np.poly1d(fit)
# y_fit = f(x)
# plt.scatter(x, y, s=0.3)
# plt.plot(x, y_fit, color='red')

if args.morestats:
    import csv
    workerid, epnum, cumulativereward, seed = [], [], [], []
    with open(args.morestats, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            cumulativereward.append(float(row['cumulativereward']))
            seed.append(float(row['seed']))
    plt.subplot(223)
    plt.title('workers episode_reward')
    plt.xlabel('Episode')
    x, y = np.array(range(len(cumulativereward))), np.array(cumulativereward)
    fit = np.polyfit(x, y, deg=1)
    f = np.poly1d(fit)
    y_fit = f(x)
    plt.scatter(x, y, s=0.5)
    plt.plot(x, y_fit, color='red')

    plt.subplot(224)
    plt.title('worker seed')
    plt.xlabel('Episode')
    x, y = np.array(range(len(seed))), np.array(seed)
    fit = np.polyfit(x, y, deg=1)
    f = np.poly1d(fit)
    y_fit = f(x)
    plt.scatter(x, y, s=0.5)
    plt.plot(x, y_fit, color='red')

if args.export:
    fname = args.export
    if not fname.endswith('png'):
        fname += '.png'
    plt.savefig(fname, bbox_inches='tight')

if not args.hide:
    plt.show()
