from rl.callbacks import *
import numpy as np
import os

class MyModelCheckpoint(Callback):
    def __init__(self, filepath, interval, verbose=0):
        super(MyModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(self.total_steps, filepath))
        self.model.save_weights(filepath, overwrite=True)
        self.model.save_memory(self.filepath)

class MoreStatsCheckpoint(Callback):
    def __init__(self, filepath):
        super(MoreStatsCheckpoint, self).__init__()
        self.filepath = filepath
        self.columns = ['workerid', 'epnum', 'cumulativereward', 'seed']
        self._create_stats_file()

    def _create_stats_file(self):
        with open(self.filepath, 'w+') as f:
            f.write(",".join(self.columns) + os.linesep)

    def _write_to_file(self, line):
        with open(self.filepath, "a+") as f:
            f.write(line + os.linesep)

    def on_step_end(self, step, logs={}):
        if logs['done']:
            # Available attributes: workerid, epnum, cumulativereward, seed
            data = [str(logs[k]) for k in self.columns]
            self._write_to_file(",".join(data))

            # if not np.isnan(logs['cumulativereward']):
            #     print("Worker %s, Ep %s: FINAL REWARD = %.3f, seed = %d" % (str(logs['workerid']), str(logs['epnum']), logs['cumulativereward'], logs['seed']))