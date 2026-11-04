""" Figure of the Jupiter-family dynamics of wmpl.Utils.OrbitClassification: how the perihelion distance and the
semi-major axis of the clones evolve, and how the fraction of clones that reached the limit builds up, which shows
whether it has settled by the end of the integration. """

from __future__ import print_function, division, absolute_import

import warnings

import numpy as np

import matplotlib.patheffects as patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from wmpl.Utils.OrbitClassification import calcBinomialInterval, JFC_DYNAMICS_A_LIMIT, JFC_DYNAMICS_P_THRESHOLD, \
    JFC_DYNAMICS_Q_LIMIT


# Ink and surface of the figure, and the one hue, for the clones that reach the limit. The band and the median
#   stand for all the clones and stay neutral.
_SURFACE = '#fcfcfb'
_INK = '#0b0b0b'
_INK_SECONDARY = '#52514e'
_GRID = '#e4e3df'
_BAND = '#c9c8c2'
_REACHED = '#eb6834'

# Top of the perihelion panel and range of the semi-major axis panel (AU). Beyond them a clone is far past the
#   limits anyway, and the panels would otherwise be scaled by the single most scattered clone
_Q_TOP = 12.0
_A_RANGE = (1.0, 200.0)

# Keeps a label readable over the marks it sits on
_HALO = [patheffects.withStroke(linewidth=3, foreground=_SURFACE)]


def reachFractionOverTime(t, q, q_limit=JFC_DYNAMICS_Q_LIMIT, confidence=0.95):
    """ The fraction of clones that have reached a perihelion limit by each output time.

    Arguments:
        t: [ndarray] Output times (yr), shape (n_times,).
        q: [ndarray] Perihelion distance (AU), shape (n_clones, n_times), as from cloneEnsemble.

    Keyword arguments:
        q_limit: [float] Perihelion limit (AU). Default JFC_DYNAMICS_Q_LIMIT.
        confidence: [float] Confidence of the interval. Default 0.95.

    Return:
        (p, low, high): [tuple of ndarray] The fraction at each time and its Wilson interval. At the last time
            it is the P of the classification.
    """

    q = np.asarray(q, dtype=float)
    n_clones = int(np.count_nonzero(np.any(~np.isnan(q), axis=1)))

    if n_clones == 0:
        raise ValueError("No clone has a state, so there is no fraction to draw.")

    count = np.cumsum(np.maximum.accumulate(q > q_limit, axis=1), axis=0)[-1]
    bounds = np.array([calcBinomialInterval(int(k), n_clones, confidence=confidence) for k in count])

    return count/n_clones, bounds[:, 0], bounds[:, 1]


def _panel(ax, elapsed, x, reached, limit, label, y_label, max_lines):
    """ One panel of a quantity of the clones: the band and median of all, and those that reach the limit. """

    # A clone that left is beyond every limit. It stays in the percentiles as a large value, off the top
    finite = np.where(np.isinf(x), 1e9, x)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        low, median, high = np.nanpercentile(finite, [5, 50, 95], axis=0)

    ax.fill_between(elapsed, low, high, color=_BAND, linewidth=0, label="5-95% of all clones")

    # Evenly through the clones, so that a run with many does not become a solid block
    rows = np.flatnonzero(reached)
    shown = rows[np.unique(np.linspace(0, len(rows) - 1, min(max_lines, len(rows))).astype(int))] if len(rows) else rows

    for k, row in enumerate(shown):
        ax.plot(elapsed, x[row], color=_REACHED, linewidth=0.7, alpha=0.55,
            label="Clones that reach q > {:g} AU".format(JFC_DYNAMICS_Q_LIMIT) if k == 0 else None)

    ax.plot(elapsed, median, color=_INK, linewidth=2.0, label="Median of all clones")

    ax.axhline(limit, color=_INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))

    # At the left, where the clones have not spread out yet
    ax.annotate(label, xy=(0.0, limit), xycoords=('axes fraction', 'data'), xytext=(4, 3),
        textcoords='offset points', ha='left', va='bottom', color=_INK_SECONDARY, fontsize=9, path_effects=_HALO)
    ax.set_ylabel(y_label, color=_INK)


def plotJupiterFamilyDynamics(result, file_path, max_lines=40):
    """ Draw the Jupiter-family dynamics of a run and save the figure.

    Arguments:
        result: [dict] Return of classifyJupiterFamilyDynamics or classifyJupiterFamilyDynamicsShober2024, called
            with return_ensemble=True.
        file_path: [str] File to save the figure to. The extension gives the format.

    Keyword arguments:
        max_lines: [int] Most clones drawn one by one in a panel. Default 40.
    """

    if result.get('ensemble') is None:
        raise ValueError("The result has no ensemble to draw: run the classification with return_ensemble=True.")

    t, q, a = result['ensemble']['t'], result['ensemble']['q'], result['ensemble']['a']

    has_state = np.any(~np.isnan(q), axis=1)
    if not np.any(has_state):
        raise ValueError("Every clone was unbound, so there is nothing to draw.")

    q, a = q[has_state], a[has_state]
    a = np.where(a < 0, np.inf, a)

    elapsed = np.abs(t)
    reached = np.any(q > JFC_DYNAMICS_Q_LIMIT, axis=1)
    s = result['assumptions']

    fig = Figure(figsize=(8.0, 9.0), facecolor=_SURFACE, layout='constrained')
    FigureCanvasAgg(fig)
    ax_q, ax_a, ax_p = fig.subplots(3, 1, sharex=True, gridspec_kw={'height_ratios': [1.1, 1.1, 0.8]})

    _panel(ax_q, elapsed, q, reached, JFC_DYNAMICS_Q_LIMIT, "q = {:g} AU".format(JFC_DYNAMICS_Q_LIMIT),
        "Perihelion distance (AU)", max_lines)
    ax_q.set_ylim(0.0, _Q_TOP)

    _panel(ax_a, elapsed, a, reached, JFC_DYNAMICS_A_LIMIT, "a = {:g} AU".format(JFC_DYNAMICS_A_LIMIT),
        "Semi-major axis (AU)", max_lines)
    ax_a.set_yscale('log')
    ax_a.set_ylim(*_A_RANGE)

    p, low, high = reachFractionOverTime(t, q, confidence=s['confidence'])

    ax_p.fill_between(elapsed, 100*low, 100*high, color=_BAND, linewidth=0,
        label="{:.0%} interval".format(s['confidence']))
    ax_p.plot(elapsed, 100*p, color=_REACHED, linewidth=2.0, label="Clones that reached q > {:g} AU so far".format(
        JFC_DYNAMICS_Q_LIMIT))
    ax_p.axhline(100*JFC_DYNAMICS_P_THRESHOLD, color=_INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)))
    ax_p.annotate("Threshold {:.0%}".format(JFC_DYNAMICS_P_THRESHOLD), xy=(0.0, 100*JFC_DYNAMICS_P_THRESHOLD),
        xycoords=('axes fraction', 'data'), xytext=(4, 3), textcoords='offset points', ha='left', va='bottom',
        color=_INK_SECONDARY, fontsize=9, path_effects=_HALO)
    ax_p.annotate("{:.1f}%".format(100*p[-1]), xy=(1.0, 100*p[-1]), xycoords=('axes fraction', 'data'), xytext=(-4, 5),
        textcoords='offset points', ha='right', va='bottom', color=_INK, fontsize=10, fontweight='bold',
        path_effects=_HALO)
    ax_p.set_ylim(0.0, 100.0)
    ax_p.set_ylabel("Clones (%)", color=_INK)
    ax_p.set_xlabel("Years before the {:s}".format("fireball" if s['mode'] == 'shober2024' else "epoch"), color=_INK)
    ax_p.set_xlim(elapsed[0], elapsed[-1])
    ax_p.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=_INK_SECONDARY, ncols=2)

    ax_q.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=_INK_SECONDARY, ncols=3)

    for ax in (ax_q, ax_a, ax_p):
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_INK_SECONDARY, labelsize=9, length=3)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(_INK_SECONDARY)

    title = "Jupiter-family dynamics: {:s}".format(result['dynamics_class'])
    shown = min(max_lines, int(np.count_nonzero(reached)))
    subtitle = ("P = {:.1f}% [{:.1f}, {:.1f}]   f_q = {:.3f}   f_a = {:.3f}   {:d} clones, {:.0f} yr, {:s}".format(
        100*result['p'], 100*result['p_low'], 100*result['p_high'], result['f_q'], result['f_a'],
        result['n_clones'], s['years'], 'setup of Shober et al. 2024' if s['mode'] == 'shober2024'
        else 'adapted mode'))
    if shown < np.count_nonzero(reached):
        subtitle += "\nOf the {:d} clones that reach the limit, {:d} are drawn one by one".format(
            int(np.count_nonzero(reached)), shown)

    fig.suptitle(title + "\n", color=_INK, fontsize=13, fontweight='bold')
    ax_q.set_title(subtitle, color=_INK_SECONDARY, fontsize=9.5, loc='left', pad=8)

    fig.savefig(file_path, dpi=150, facecolor=_SURFACE)
