from typing import Tuple

import h5py
import numpy as np
import pandas as pd

from constants import *


def load_data(
    path_to_training_data: str, path_to_training_target: str, n_data=4400
) -> Tuple[np.ndarray, np.ndarray]:
    file = h5py.File(path_to_training_data, "r")
    data = file["data"]

    """
    - 0: sample index
    - 1: subject index
    - 2 to 9001: Abdominal belt
    - 9002 to 18001: Airflow
    - 18002 to 27001: PPG (Photoplethysmogram)
    - 27002 to 36001: Thoracic belt
    - 36002 to 45001: Snoring indicator
    - 45002 to 54001: SPO2
    - 54002 to 63001: C4-A1
    - 63002 to 72001:O2-A1
    """

    x = data[:n_data, 2:]
    x = x.reshape(n_data, N_signals, -1)
    x = np.transpose(x, (0, 2, 1))
    x = x.reshape(n_data * 9000 // window_size, window_size, N_signals)
    x = np.transpose(x, (0, 2, 1))
    mask = np.array(pd.read_csv(path_to_training_target))[:n_data, 1:].flatten()

    return x, mask
