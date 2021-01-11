import numpy as np
import random
from collections import Counter
import datetime as dt
import inspect

def countRepeats(l):
    return len(l) - len(set(l))

length = 30000
size = 128
test_cases = 30000

fns = []
fns.append(lambda: np.random.randint(0, length, size=size))
fns.append(lambda: random.sample(xrange(length), size))
fns.append(lambda: np.random.choice(length, size, False))

for fn in fns:
    print(inspect.getsource(fn))
    now = dt.datetime.now()
    lists = [fn() for _ in range(test_cases)]
    elapsed = (dt.datetime.now() - now).total_seconds()
    repeats = [countRepeats(l) for l in lists]
    counts = Counter(sorted(repeats))
    print(sorted(counts.items()))
    print("Took %.1f seconds." % elapsed)
    print("")
