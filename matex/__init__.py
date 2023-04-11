from dataclasses import dataclass

import numpy as np


@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    next_state: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: np.ndarray
