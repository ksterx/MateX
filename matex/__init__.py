from typing import NamedTuple

import numpy as np


class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: np.ndarray
