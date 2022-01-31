from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt

from constants import *


def visualize_signal(
    X: np.ndarray,
    indexes: List[int],
    window_size: int = 100,
    title: Union[None, str] = None,
) -> None:
    fig, axs = plt.subplots(N_signals, len(indexes), sharex=True, sharey=True)
    for i in range(N_signals):
        for j, ind in enumerate(indexes):
            axs[i][j].plot(np.arange(0, window_size * 10, 10), X[ind, i, :])
            axs[i][j].set_ylim(-5, 5)
        axs[i][0].set_ylabel(SIGNALS_NAME[i])
    for j in range(len(indexes)):
        axs[-1, j].set_xlabel("Time (ms)")
    plt.xlim(0, window_size)

    if title is not None:
        fig.suptitle(title)
