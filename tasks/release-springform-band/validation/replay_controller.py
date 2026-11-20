"""Use as controller.py in a copy of the task to replay a recorded action sequence."""
import gzip
import json
from pathlib import Path

import numpy as np


class Controller:
    def __init__(self, observer):
        with gzip.open(Path(__file__).with_name('actions.json.gz'), 'rt') as source:
            self.actions = json.load(source)

    def action(self, t):
        return np.asarray(self.actions[round(t * 20) - 1], dtype=float)
