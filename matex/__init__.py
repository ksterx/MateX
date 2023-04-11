from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: np.ndarray
    terminated: np.ndarray
