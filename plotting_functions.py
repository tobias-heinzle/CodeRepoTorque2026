from typing import Optional, Callable
import warnings

from matplotlib.collections import LineCollection
import numpy as np

from scipy.stats import Normal

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import colormaps

from jax import Array

from baum_welch_jax import HiddenMarkovParameters
import pandas as pd

from util import normalize_timestamp

def plot_hmm_params(
        hmm: HiddenMarkovParameters, 
        plot_mu: bool=True, 
        with_numbers: bool=False, 
        cmap: str='viridis',
        ax: Optional[np.array] = None
        ) -> tuple[Figure | None, np.ndarray[Axes]]:
    n, m = hmm.O.shape

    cm = colormaps.get_cmap(cmap)

    if hmm.is_log:
        hmm = hmm.to_prob()

    if plot_mu:
        if hmm.mu.ndim == 1:
            n_mu = 1
            _mu = hmm.mu[:,None]
        else:
            n_mu = hmm.mu.shape[0]
            _mu = hmm.mu.T

        width_ratios = (n, m, n_mu)
        
    else:
        width_ratios = (n, m)
    
    if ax is None:
        fig, ax = plt.subplots(1, len(width_ratios), width_ratios=width_ratios, constrained_layout=True)
        fig.set_size_inches((2.3, 2.3))
    else:
        fig = None

    ax[0].matshow(hmm.T, cmap=cmap)
    ax[0].set_title("T")

    ax[1].matshow(hmm.O, cmap=cmap)
    ax[1].set_title("O")


    if plot_mu:
        ax[2].matshow(_mu, cmap=cmap)
        ax[2].set_title(r"$\mu$")
    
    if with_numbers:
        for idx, mat in enumerate((hmm.T, hmm.O, hmm.mu)):
            if idx == 2 and not plot_mu:
                break
            for (i, j), val in np.ndenumerate(mat):
                if val >= 1e-10:
                    fill_str = f"{val:.1g}" if val < 0.9 else f"{val:.5f}"
                    fill_str = fill_str.lstrip('0')
                    if val == 1.0:
                        fill_str = '1'
                    ax[idx].text(
                        j, i, fill_str, 
                        ha="center", va="center", 
                        fontsize=mpl.rcParams['xtick.labelsize'] - 1,  # same as ticks
                        color=cm(0.0) if val > mat.max()/2 else cm(1.0)
                )

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    
    return fig, ax



def plot_array_hist_over_time(
        distr_data: np.ndarray, 
        color: str, 
        bins: np.ndarray, 
        name=None, 
        interval: int = 30) -> tuple[Figure, Axes]:


    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(20,-35, 0)
    ax.set_box_aspect([1, 5, 1])

    yticks = np.arange(len(distr_data) // interval)


    ks = range(len(yticks))

    for window_start, k in zip(yticks, ks):
        window_start = interval * window_start

        ys, xs = np.histogram(distr_data[window_start: window_start + interval], bins, density=True)

 
        # Plot the bar graph given by xs and ys on the plane y=k
        ax.bar(xs[1:], ys, width=xs[1] - xs[0], zs=k, zdir='y', color=color, alpha=0.75, edgecolor='black')

    ax.set_xlabel('\nIndicator value')
    ax.set_zlabel('\nDensity')
    ax.set_ylabel('\n\n\n\n\nMonth')

    # On the y-axis let's only label the discrete values that we have data for.
    ax.set_yticks(ks[::2], ['\n' + str(y) for y in yticks[::2]])
    ax.set_xticks(np.linspace(0, 0.6, 4, endpoint=True))
    ax.set_title('Evolution of the 7 day average of observations over time')

    if name is not None:
        plt.savefig(f'plots/{name}.pdf', dpi=300, bbox_inches=mpl.transforms.Bbox(np.array([[4,4.5], [16.5,10]])))

    return fig, ax


def plot_residual_sequences(data: list[pd.DataFrame]) -> tuple[Figure, Axes]:
    window = 6 * 25 * 7
    anom_count = 0

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    for df in data:
        x = normalize_timestamp(df.time_stamp[window-1:])
        
        y = np.convolve(df.residual, np.ones(window)/window, mode='valid')
        label = df['label'].iloc[0]
        color = 'blue'
        alpha = 0.25
        if label == 'anomaly':
            color = 'red'
            alpha = 0.25 + anom_count/3
            anom_count += 1
        ax.step(x, y, color=color, alpha=alpha)
    ax.grid()
    ax.set_title('Residuals')
    return fig, ax


def plot_bin_borders_onto_residuals(data: list[pd.DataFrame], bin_borders: np.ndarray) -> tuple[Figure, np.ndarray[Axes]]:
    fig, ax = plt.subplots(2)
    fig.set_size_inches(8,5)

    window = 6 * 25 * 7
    linewidth = 2.0


    x_max = [0]
    min_len = 1e20
    for df in data:
        label = str(df["label"].iloc[0])
        
        xs = normalize_timestamp(df.time_stamp[window-1:])
        
        ys = np.convolve(df.residual, np.ones(window)/window, mode='valid')

        if len(xs) > len(x_max):
            x_max = xs.to_numpy()
        if len(xs) < min_len:
            min_len = len(xs)

        if label == "normal":
            ax[0].plot(xs, ys, alpha=0.5, color="blue", linewidth=linewidth)
        else:
            ax[1].plot(xs, ys, alpha=0.5, color="red", linewidth=linewidth)
    for _ax in ax:
        _ax.set_xlim(x_max[0], x_max[-1])
        _ax.set_ylim(0,0.4)
        _ax.hlines(bin_borders, x_max[0], x_max[-1], colors="black", linestyles='dashed', alpha=0.5)
    ax[0].set_title("Normal behaviour")
    ax[1].set_title("Abnormal behaviour")

    return fig, ax


def plot_daily_time_series(daily_time_series: list[np.ndarray]) -> tuple[Figure, np.ndarray[Axes]]:
    fig, ax = plt.subplots(2)
    fig.set_size_inches(8,5)

    linewidth = 2.0

    for seq in daily_time_series:
        if np.any(seq >= 0.99):
            ax[1].plot(seq, alpha=0.5, color="red", linewidth=linewidth)
        else:
            ax[0].plot(seq, alpha=0.5, color="blue", linewidth=linewidth)

            pass
    for _ax in ax:
        _ax.grid()
        _ax.set_ylim(0,1.0)

    ax[0].set_title("Normal behaviour")
    ax[1].set_title("Abnormal behaviour")

    return fig, ax

def plot_time_to_failure_distribution(years: int = 30, mu: float = 3.972585041839995, sigma: float = 1.2650367166361958) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(2.3, 1.6))

    days = np.arange(1, 365*years + 1)

    _cdf = np.array(Normal(mu=mu, sigma=sigma).cdf(np.log(days/365)))

    ax.plot(days / 365, _cdf, 'k', label='Lognormal model')
    ax.grid()
    ax.set_title('Lognormal failure distribution.')
    ax.set_xlabel('Time [years]')
    ax.set_ylabel('Failure probability')
    ax.legend()

    return fig, ax

def plot_fit(
        arr: Array, 
        final_state_arrival_distribution: Callable[[Array, int], Array],
        ax: Optional[np.ndarray] = None,
        color_truth: tuple | str = 'darkgrey',
        color_fitted: tuple | str = plt.cm.tab20b.colors[1],
        mu: float = 3.972585041839995, 
        sigma: float = 1.2650367166361958) -> tuple[Figure | None, np.ndarray]:
    years = 30

    days = np.arange(1, 365*years + 1)


    _cdf = np.array(Normal(mu=mu, sigma=sigma).cdf(np.log(days/365)))
    _, arrival = final_state_arrival_distribution(arr, 365 * years)

    if ax is None:
        fig, ax = plt.subplots(figsize=(2.3, 2.3), constrained_layout=True)
    else:
        fig = None

    ax.plot(days / 365, _cdf, label='$F(t)$', color=color_truth)
    ax.plot(days / 365, arrival, linestyle='dashed', label='$\\mathbb{P}[\\tau \\leq t ~|~ \\mathbf{T}]$', color=color_fitted)
    ax.grid()
    ax.set_ylabel('Failure probability')
    ax.set_title('Time-to-failure distribution')
    ax.set_xlabel('$t$ (years)')    
    ax.legend()
    
    return fig, ax
