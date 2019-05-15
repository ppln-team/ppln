from collections import defaultdict

import numpy as np


class LogBuffer(object):
    def __init__(self):
        self.value_history = defaultdict(list)
        self.n_history = defaultdict(list)
        self.output = dict()

    def clear(self):
        self.value_history.clear()
        self.n_history.clear()
        self.output.clear()

    def update(self, values, count=1):
        assert isinstance(values, dict)
        for key, value in values.items():
            self.value_history[key].append(value)
            self.n_history[key].append(count)

    def average(self, n=0):
        """Average latest n values or all values"""
        assert n >= 0
        for key in self.value_history:
            values = np.array(self.value_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
