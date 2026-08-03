from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MultipleLocator

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Utils.extraction_module import StockDataWarehouse

DOE_PALETTE = ("#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974")


class StockTimeSeriesPlotter:
    """Builds and saves visualizations of the temporal behavior of stocks held in a StockDataWarehouse."""

    def __init__(self, warehouse: StockDataWarehouse):
        self.warehouse = warehouse

    def plot_price_series(self, normalize: bool = True, log_scale: bool = True, figsize: tuple = (12, 9)) -> plt.Figure:
        """
        Plot the historical price series of every stock in the warehouse.

        Parameters:
        normalize (bool): If True, rebase every series to 100 at its first date so that
                           stocks with very different price levels remain comparable.
        log_scale (bool): If True, add a second panel with the y-axis in logarithmic scale,
                           in the same figure, useful to compare relative growth across assets.
        figsize (tuple): Size of the matplotlib figure.

        Returns:
        plt.Figure: The generated figure.
        """
        panel = self.warehouse.price_panel()
        if normalize:
            panel = panel.div(panel.iloc[0]).mul(100)

        scales = ["linear", "log"] if log_scale else ["linear"]
        fig, axes = plt.subplots(len(scales), 1, figsize=figsize, sharex=True)
        axes = [axes] if len(scales) == 1 else axes

        ylabel = "Preco indexado (base 100)" if normalize else "Preco ajustado"
        for ax, scale in zip(axes, scales):
            for ticker in panel.columns:
                ax.plot(panel.index, panel[ticker], label=ticker)
            ax.set_yscale(scale)
            ax.set_ylabel(f"{ylabel} (escala {scale})")

        axes[0].set_title("Comportamento temporal dos ativos")
        axes[0].legend(loc="upper left", ncol=2, fontsize="small")
        axes[-1].set_xlabel("Data")
        fig.tight_layout()
        return fig

    def save_figure(self, fig: plt.Figure, path: str) -> None:
        """Save a figure to disk, creating parent directories if needed."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)



def _looks_like_simplex(points: np.ndarray, atol: float = 1e-3) -> bool:
    """Detect whether 3-column points lie on the simplex {x >= 0, sum(x) = 1}."""
    return bool(np.all(points >= -atol) and np.allclose(points.sum(axis=1), 1.0, atol=atol))


def _draw_simplex_boundary(ax, color: str = "0.25", lw: float = 1.3) -> None:
    """Draw the triangular boundary of the unit simplex spanned by the 3 basis vectors."""
    vertices = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], color=color, lw=lw, zorder=1)


def _style_3d_axes(ax, elev: float, azim: float, roll: float,
                    tick_step: Optional[float] = 0.2,
                    box_aspect: Optional[Tuple[float, float, float]] = (1, 1, 1)) -> None:
    """
    Apply a clean, consistent look to a 3D axes: light panes, fixed view angle,
    and (optionally) a cubic aspect ratio and fixed tick spacing.

    box_aspect and tick_step assume a shared [0, 1]-like domain across axes
    (the DoE/simplex case); pass None for either to leave matplotlib's
    automatic aspect/ticks in place instead (e.g. for an arbitrary function's
    input/output domain, where forcing a cube would distort the plot).
    """
    if box_aspect is not None:
        ax.set_box_aspect(box_aspect)
    ax.view_init(elev=elev, azim=azim, roll=roll)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_alpha(1.0)
        axis.pane.set_edgecolor("0.85")
        if tick_step is not None:
            axis.set_major_locator(MultipleLocator(tick_step))
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=9)


def _style_log_colorbar(cbar, label: Optional[str] = None) -> None:
    """
    Clean up a log-scaled colorbar: matplotlib's default LogLocator, given a
    range spanning less than a couple of decades (typical for e.g. a posterior
    standard deviation), places ticks at whatever values evenly subdivide the
    range in log-space — arbitrary numbers like 4.11772e-07, not round ones.
    Restricting ticks to 1x/2x/5x per decade keeps them readable, and the
    label gets an explicit "(escala log)" suffix so the scale itself is never
    ambiguous from the numbers alone.
    """
    cbar.ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    cbar.ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs="auto"))
    cbar.ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    if label:
        cbar.set_label(f"{label} (escala log)", fontsize=10)


def plot_doe_3d(points,
                 labels: Sequence[str] = ("x1", "x2", "x3"),
                 title: Optional[str] = None,
                 ax=None,
                 elev: float = 20,
                 azim: float = 45,
                 roll: float = 0,
                 color: str = DOE_PALETTE[0],
                 s: float = 35,
                 alpha: float = 0.75,
                 figsize: Tuple[float, float] = (7, 6.5),
                 show_simplex: Optional[bool] = None,
                 show: bool = True):
    """
    Plot a design of experiments in 3D.

    points: array (n_samples, dim) as returned by a SamplerShell. Only the
        first three columns are used; raises if dim < 3.
    labels: axis names.
    title: optional plot title.
    ax: existing 3D axes to draw on (e.g. one subplot of a comparison grid);
        a new figure and axes are created when omitted.
    elev, azim, roll: viewing angle in degrees, forwarded to Axes3D.view_init,
        letting the plot be rotated to whichever perspective best exposes the
        sample's structure. Defaults are tuned for points constrained to a
        simplex (e.g. portfolio weights), where they render it as a
        near-frontal triangle instead of the edge-on line the default
        matplotlib angle produces.
    color, s, alpha: scatter marker color, size and opacity.
    figsize: size of the created figure; ignored when ax is given.
    show_simplex: whether to draw the simplex boundary triangle. When None,
        it is drawn automatically if the points sum to 1 along every row.
    show: whether to call plt.show() after building the figure; set to False
        when composing this plot inside a larger figure or saving headlessly.

    Returns: (fig, ax).
    """
    points = np.asarray(points)
    if points.shape[1] < 3:
        raise ValueError(f"sao necessarias ao menos 3 dimensoes (recebido dim={points.shape[1]})")

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if show_simplex is None:
        show_simplex = _looks_like_simplex(points[:, :3])
    if show_simplex:
        _draw_simplex_boundary(ax)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               s=s, alpha=alpha, color=color, edgecolor="white", linewidth=0.4, zorder=2)

    ax.set_xlabel(labels[0], labelpad=10, fontsize=11)
    ax.set_ylabel(labels[1], labelpad=10, fontsize=11)
    ax.set_zlabel(labels[2], labelpad=6, fontsize=11)

    pad = 0.05
    lo = min(0.0, points[:, :3].min())
    hi = max(1.0, points[:, :3].max())
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_zlim(lo - pad, hi + pad)

    if title:
        ax.set_title(title, fontsize=13, pad=12)

    _style_3d_axes(ax, elev=elev, azim=azim, roll=roll)

    if created_fig:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_doe_comparison_3d(samples: Dict[str, np.ndarray],
                            labels: Sequence[str] = ("x1", "x2", "x3"),
                            suptitle: Optional[str] = None,
                            elev: float = 20,
                            azim: float = 45,
                            roll: float = 0,
                            colors: Optional[Sequence[str]] = None,
                            s: float = 35,
                            alpha: float = 0.75,
                            panel_size: Tuple[float, float] = (6, 6),
                            show: bool = True):
    """
    Plot several designs of experiments side by side in 3D, sharing axis
    limits and viewing angle so their coverage of the domain can be
    compared directly (e.g. LHS vs. Sobol over the same simplex).

    samples: mapping from a display name (used as each panel's title) to an
        (n_samples, dim) array of points.
    labels, elev, azim, roll, s, alpha: forwarded to plot_doe_3d for every panel.
    colors: one color per panel; defaults to DOE_PALETTE, cycled if needed.
    panel_size: (width, height) of each individual panel.
    show: whether to call plt.show() after building the figure.

    Returns: fig.
    """
    names = list(samples.keys())
    n = len(names)
    colors = colors or DOE_PALETTE

    fig = plt.figure(figsize=(panel_size[0] * n, panel_size[1]))
    for i, name in enumerate(names):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        plot_doe_3d(samples[name], labels=labels, title=name, ax=ax,
                    elev=elev, azim=azim, roll=roll,
                    color=colors[i % len(colors)], s=s, alpha=alpha, show=False)

    fig.tight_layout(w_pad=4.0)
    # tight_layout underestimates how much room the leftmost panel's z-label
    # needs (Axes3D does not report it through the usual bbox machinery), and
    # a suptitle needs headroom reserved explicitly for the same reason.
    fig.subplots_adjust(left=0.06, right=0.96, top=0.86 if suptitle else 0.94)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.98)
    if show:
        plt.show()
    return fig


def plot_metric_3d(points,
                    values,
                    labels: Sequence[str] = ("x1", "x2", "x3"),
                    title: Optional[str] = None,
                    ax=None,
                    elev: float = 20,
                    azim: float = 45,
                    roll: float = 0,
                    cmap: str = "viridis",
                    s: float = 40,
                    alpha: float = 0.85,
                    figsize: Tuple[float, float] = (7.5, 6.5),
                    show_simplex: Optional[bool] = None,
                    vmin: Optional[float] = None,
                    vmax: Optional[float] = None,
                    log_scale: bool = False,
                    colorbar_label: Optional[str] = None,
                    show_colorbar: bool = True,
                    show: bool = True):
    """
    Plot a scalar metric evaluated at a set of 3D points, coloring each point
    by its value instead of scattering them uniformly. Meant for anything
    evaluated over a DoE or the portfolio simplex: a financial measure
    (CVaR, STARR, ...), or later a surrogate's posterior mean/std.

    points: array (n_samples, dim); only the first three columns are used.
    values: array (n_samples,), one scalar per row of points.
    labels: axis names.
    title: optional plot title.
    ax: existing 3D axes to draw on; a new figure/axes are created when omitted.
    elev, azim, roll: viewing angle in degrees, see plot_doe_3d.
    cmap: colormap mapping values to marker color.
    s, alpha: scatter marker size and opacity.
    figsize: size of the created figure; ignored when ax is given.
    show_simplex: whether to draw the simplex boundary triangle; auto-detected
        from points when None, as in plot_doe_3d.
    vmin, vmax: color scale limits; default to this call's own data range.
        Pass explicitly to keep the color scale comparable across separate
        calls (plot_metric_comparison_3d does this automatically).
    log_scale: color on a logarithmic scale instead of linear; values must be
        strictly positive. Useful for quantities like a posterior standard
        deviation, which are always positive and often vary over orders of
        magnitude — on a linear scale a handful of high-uncertainty points
        wash out any variation in the (much larger) low-uncertainty bulk.
    colorbar_label: label drawn alongside the colorbar.
    show_colorbar: whether to attach a colorbar to this axes.
    show: whether to call plt.show() after building the figure.

    Returns: (fig, ax, scatter) — scatter is the mappable backing the colorbar.
    """
    points = np.asarray(points)
    values = np.asarray(values)
    if points.shape[1] < 3:
        raise ValueError(f"sao necessarias ao menos 3 dimensoes (recebido dim={points.shape[1]})")
    assert values.shape[0] == points.shape[0], \
        f"values must have one entry per row of points ({points.shape[0]}, current is {values.shape[0]})"
    if log_scale:
        assert np.all(values > 0), "log_scale=True requires every value to be strictly positive"

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if show_simplex is None:
        show_simplex = _looks_like_simplex(points[:, :3])
    if show_simplex:
        _draw_simplex_boundary(ax)

    color_kwargs = {"norm": LogNorm(vmin=vmin or values.min(), vmax=vmax or values.max())} \
        if log_scale else {"vmin": vmin, "vmax": vmax}
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                          c=values, cmap=cmap, **color_kwargs,
                          s=s, alpha=alpha, edgecolor="white", linewidth=0.4, zorder=2)

    ax.set_xlabel(labels[0], labelpad=10, fontsize=11)
    ax.set_ylabel(labels[1], labelpad=10, fontsize=11)
    ax.set_zlabel(labels[2], labelpad=6, fontsize=11)

    pad = 0.05
    lo = min(0.0, points[:, :3].min())
    hi = max(1.0, points[:, :3].max())
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_zlim(lo - pad, hi + pad)

    if title:
        ax.set_title(title, fontsize=13, pad=12)

    _style_3d_axes(ax, elev=elev, azim=azim, roll=roll)

    if show_colorbar:
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.65, pad=0.1)
        if log_scale:
            _style_log_colorbar(cbar, colorbar_label)
        elif colorbar_label:
            cbar.set_label(colorbar_label, fontsize=10)

    if created_fig:
        fig.tight_layout()
        # same Axes3D z-label headroom issue as plot_doe_comparison_3d, worse
        # here since the colorbar eats into tight_layout's already-tight margin
        fig.subplots_adjust(left=0.1)
    if show:
        plt.show()
    return fig, ax, scatter


def plot_metric_comparison_3d(points,
                               metrics: Dict[str, np.ndarray],
                               labels: Sequence[str] = ("x1", "x2", "x3"),
                               suptitle: Optional[str] = None,
                               elev: float = 20,
                               azim: float = 45,
                               roll: float = 0,
                               cmap: str = "viridis",
                               s: float = 40,
                               alpha: float = 0.85,
                               panel_size: Tuple[float, float] = (6.5, 6),
                               shared_scale: bool = False,
                               log_scale: Any = False,
                               show: bool = True):
    """
    Plot several scalar metrics evaluated at the same set of 3D points side
    by side (e.g. CVaR vs. STARR over the same DoE), sharing axis limits and
    viewing angle so their spatial pattern over the domain can be compared.

    points: array (n_samples, dim), shared by every panel.
    metrics: mapping from a display name (used as each panel's title and
        colorbar label) to an (n_samples,) array of values for that metric.
    labels, elev, azim, roll, cmap, s, alpha: forwarded to plot_metric_3d.
    panel_size: (width, height) of each individual panel.
    shared_scale: if True, every panel uses the same color scale (the min/max
        across all metrics combined). Only meaningful when the metrics share
        comparable units/magnitude; otherwise it saturates the colors of
        whichever metric has the smaller range.
    log_scale: forwarded to plot_metric_3d's log_scale. Either a single bool
        applied to every panel, or a {name: bool} dict for mixed cases (e.g.
        a posterior mean, which can be negative, plotted next to its always-
        positive standard deviation, which usually benefits from log_scale=True).
    show: whether to call plt.show() after building the figure.

    Returns: fig.
    """
    names = list(metrics.keys())
    n = len(names)

    vmin = vmax = None
    if shared_scale:
        all_values = np.concatenate([np.asarray(metrics[name]) for name in names])
        vmin, vmax = float(all_values.min()), float(all_values.max())

    fig = plt.figure(figsize=(panel_size[0] * n, panel_size[1]))
    for i, name in enumerate(names):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        panel_log_scale = log_scale.get(name, False) if isinstance(log_scale, dict) else log_scale
        plot_metric_3d(points, metrics[name], labels=labels, title=name, ax=ax,
                        elev=elev, azim=azim, roll=roll, cmap=cmap, s=s, alpha=alpha,
                        vmin=vmin, vmax=vmax, log_scale=panel_log_scale, colorbar_label=name, show=False)

    fig.tight_layout(w_pad=4.0)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.86 if suptitle else 0.94)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.98)
    if show:
        plt.show()
    return fig


def _gp_prior_samples(kernel: Any, X: np.ndarray, n_samples: int,
                       seed: Optional[int], jitter: float = 1e-8) -> np.ndarray:
    """Draws n_samples function values at X (n_points, n_dims) from a zero-mean GP prior with the given kernel."""
    K = kernel(X)
    K[np.diag_indices_from(K)] += jitter
    L = np.linalg.cholesky(K)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((X.shape[0], n_samples))
    return L @ z


def plot_gp_prior_1d(kernel: Any,
                      bounds: Tuple[float, float] = (-3, 3),
                      n_points: int = 200,
                      n_samples: int = 5,
                      seed: Optional[int] = 0,
                      title: Optional[str] = None,
                      ax=None,
                      figsize: Tuple[float, float] = (7, 5),
                      colors: Optional[Sequence[str]] = None,
                      show: bool = True):
    """
    Plot functions sampled from a zero-mean Gaussian process prior with the
    given kernel, over a single input dimension. Useful to see, before ever
    fitting data, what kind of landscape a candidate kernel is capable of
    representing (smooth and mean-reverting, trending, etc.).

    kernel: a scikit-learn Kernel (or any object exposing kernel(X) -> (n, n)
        covariance and kernel.diag(X) -> (n,) variance), built for 1 input
        dimension (e.g. SurrogateKernels.Plain(n_dims=1)). Evaluated exactly
        as constructed: any trend or noise term it includes is part of the draw.
    bounds: (low, high) of the 1D input grid.
    n_points: resolution of the input grid.
    n_samples: number of independent functions drawn from the prior.
    seed: fixes the draw.
    title: optional plot title.
    ax: existing 2D axes to draw on; a new figure/axes are created when omitted.
    figsize: size of the created figure; ignored when ax is given.
    colors: one color per sampled function; defaults to DOE_PALETTE, cycled.
    show: whether to call plt.show() after building the figure.

    Returns: (fig, ax).
    """
    x = np.linspace(bounds[0], bounds[1], n_points)
    X = x[:, None]
    samples = _gp_prior_samples(kernel, X, n_samples, seed)
    std = np.sqrt(np.clip(kernel.diag(X), 0, None))
    colors = colors or DOE_PALETTE

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.fill_between(x, -2 * std, 2 * std, color="0.88", label="prior ±2σ", zorder=1)
    for i in range(n_samples):
        ax.plot(x, samples[:, i], color=colors[i % len(colors)], lw=1.6, alpha=0.9, zorder=2)
    ax.axhline(0, color="0.4", lw=0.8, linestyle=":", zorder=1)

    ax.set_xlabel("x", fontsize=11)
    ax.set_ylabel("f(x)", fontsize=11)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title, fontsize=13, pad=10)

    if created_fig:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_gp_prior_surface_2d(kernel: Any,
                              bounds: Tuple[Tuple[float, float], Tuple[float, float]] = ((-3, 3), (-3, 3)),
                              n_points: int = 45,
                              seed: Optional[int] = 0,
                              elev: float = 25,
                              azim: float = -50,
                              roll: float = 0,
                              title: Optional[str] = None,
                              ax=None,
                              figsize: Tuple[float, float] = (7, 6.5),
                              cmap: str = "viridis",
                              show: bool = True):
    """
    Plot a single function sampled from a zero-mean Gaussian process prior
    with the given kernel, over a 2D input grid, as a 3D surface. The 2-input
    case matches this project's own use of these kernels: fitting a surrogate
    over 3-asset portfolios projects them onto a 2D (Helmert) subspace, so this
    is literally the kind of landscape SurrogateGPR would be asked to fit.

    kernel: a scikit-learn Kernel (or any object exposing kernel(X) -> (n, n)
        covariance), built for 2 input dimensions (e.g.
        SurrogateKernels.Plain(n_dims=2)). Evaluated exactly as constructed.
    bounds: ((x1_low, x1_high), (x2_low, x2_high)) of the input grid.
    n_points: grid resolution per axis (total evaluated points = n_points**2;
        keep this modest, since the covariance matrix scales as
        (n_points**2)**2).
    seed: fixes the draw.
    elev, azim, roll: viewing angle in degrees, forwarded to Axes3D.view_init.
    title: optional plot title.
    ax: existing 3D axes to draw on; a new figure/axes are created when omitted.
    figsize: size of the created figure; ignored when ax is given.
    cmap: colormap for the surface.
    show: whether to call plt.show() after building the figure.

    Returns: (fig, ax).
    """
    x1 = np.linspace(bounds[0][0], bounds[0][1], n_points)
    x2 = np.linspace(bounds[1][0], bounds[1][1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.column_stack([X1.ravel(), X2.ravel()])
    sample = _gp_prior_samples(kernel, X, 1, seed)[:, 0].reshape(X1.shape)

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot_surface(X1, X2, sample, cmap=cmap, linewidth=0, antialiased=True, alpha=0.95)

    ax.set_xlabel("x1", labelpad=8, fontsize=11)
    ax.set_ylabel("x2", labelpad=8, fontsize=11)
    ax.set_zlabel("f(x1, x2)", labelpad=6, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, pad=12)

    _style_3d_axes(ax, elev=elev, azim=azim, roll=roll, tick_step=None, box_aspect=None)

    if created_fig:
        fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def plot_gp_prior_comparison(kernels: Dict[str, Any],
                              n_dims: int = 1,
                              suptitle: Optional[str] = None,
                              panel_size: Tuple[float, float] = (6, 5.5),
                              seed: Optional[int] = 0,
                              show: bool = True,
                              **kwargs):
    """
    Plot, side by side, functions sampled from the prior of several kernels,
    making their differing assumptions (smoothness, trend, anisotropy)
    visually comparable.

    kernels: mapping from a display name (used as each panel's title) to a
        kernel built for the given n_dims (e.g. SurrogateKernels.Plain(n_dims)).
    n_dims: 1 draws sample paths via plot_gp_prior_1d, one 2D line panel per
        kernel, sharing y-limits. 2 draws a sampled surface via
        plot_gp_prior_surface_2d, one 3D panel per kernel, sharing z-limits.
    panel_size: (width, height) of each individual panel.
    seed: shared across every panel, so panels differ only by kernel, not by
        which random draw was taken.
    kwargs: forwarded to plot_gp_prior_1d (bounds, n_points, n_samples, colors)
        when n_dims=1, or to plot_gp_prior_surface_2d (bounds, n_points, cmap,
        elev, azim, roll) when n_dims=2.
    show: whether to call plt.show() after building the figure.

    Returns: fig.
    """
    if n_dims not in (1, 2):
        raise ValueError(f"n_dims must be 1 or 2 (current is {n_dims})")

    names = list(kernels.keys())
    n = len(names)
    fig = plt.figure(figsize=(panel_size[0] * n, panel_size[1]))

    axes = []
    for i, name in enumerate(names):
        if n_dims == 1:
            ax = fig.add_subplot(1, n, i + 1)
            plot_gp_prior_1d(kernels[name], title=name, ax=ax, seed=seed, show=False, **kwargs)
        else:
            ax = fig.add_subplot(1, n, i + 1, projection="3d")
            plot_gp_prior_surface_2d(kernels[name], title=name, ax=ax, seed=seed, show=False, **kwargs)
        axes.append(ax)

    if n_dims == 1:
        ylo = min(ax.get_ylim()[0] for ax in axes)
        yhi = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(ylo, yhi)
        fig.tight_layout()
        if suptitle:
            fig.subplots_adjust(top=0.85)
            fig.suptitle(suptitle, fontsize=14, y=0.98)
    else:
        zlo = min(ax.get_zlim()[0] for ax in axes)
        zhi = max(ax.get_zlim()[1] for ax in axes)
        for ax in axes:
            ax.set_zlim(zlo, zhi)
        fig.tight_layout(w_pad=4.0)
        fig.subplots_adjust(left=0.06, right=0.96, top=0.86 if suptitle else 0.94)
        if suptitle:
            fig.suptitle(suptitle, fontsize=14, y=0.98)
    if show:
        plt.show()
    return fig


def _simplex_grid_2d(n_points: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Meshgrid over [0,1]x[0,1], plus a mask for the region outside {wi>=0, wj>=0, wi+wj<=1}."""
    t = np.linspace(0, 1, n_points)
    Wi, Wj = np.meshgrid(t, t)
    mask = (Wi + Wj) > 1.0 + 1e-9
    return Wi, Wj, mask


def _draw_simplex_boundary_2d(ax, color: str = "0.25", lw: float = 1.3) -> None:
    """Draw the right-triangle boundary {wi>=0, wj>=0, wi+wj<=1} of a 2D simplex slice."""
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color=color, lw=lw, zorder=3)


def plot_simplex_slices_2d(values_fn: Any,
                            labels: Sequence[str] = ("x1", "x2", "x3"),
                            n_points: int = 60,
                            cmap: str = "viridis",
                            panel_size: Tuple[float, float] = (5.5, 5),
                            suptitle: Optional[str] = None,
                            colorbar_label: Optional[str] = None,
                            vmin: Optional[float] = None,
                            vmax: Optional[float] = None,
                            log_scale: bool = False,
                            overlay_points: Optional[np.ndarray] = None,
                            show: bool = True):
    """
    Plot a scalar function of 3-asset portfolio weights as three flat 2D
    slices of the simplex, one per choice of which asset is left implicit
    (w_k = 1 - w_i - w_j). A 3-asset simplex is intrinsically 2-dimensional,
    but no single pair of the 3 assets is a privileged choice of plot axes;
    showing all three pairings gives a complete, undistorted view (no 3D
    perspective, no occlusion) without arbitrarily favoring one asset.

    values_fn: callable, values_fn(W) -> (n,) array of scalars for an
        (n, 3) array of full simplex weights W (e.g. a fitted surrogate's
        posterior mean or std via `lambda W: surrogate.predict(W, return_std=True)[0]`,
        or the true metric itself).
    labels: the 3 asset names, used as axis labels and to name each panel's
        implicit (dropped) asset.
    n_points: grid resolution along each 2D axis.
    cmap: colormap for the filled contours.
    panel_size: (width, height) of each individual panel.
    suptitle: optional figure-level title.
    colorbar_label: label for the shared colorbar.
    vmin, vmax: color scale limits; default to the combined range across all
        three panels, so the same color always means the same value on every
        panel.
    log_scale: color on a logarithmic scale instead of linear; requires every
        evaluated value to be strictly positive. See plot_metric_3d — same
        rationale, useful for a strictly-positive, wide-dynamic-range surface
        such as a posterior standard deviation.
    overlay_points: optional (n, 3) array of simplex points (e.g. the
        training DoE) scattered on top of each panel, using that panel's pair
        of axes; useful to relate the surface to where data was observed.
    show: whether to call plt.show() after building the figure.

    Returns: fig.
    """
    Wi, Wj, mask = _simplex_grid_2d(n_points)
    combos = [(1, 2, 0), (0, 2, 1), (0, 1, 2)]  # (axis i, axis j, implicit/dropped k)

    grids = []
    for i, j, k in combos:
        W = np.zeros((Wi.size, 3))
        W[:, i] = Wi.ravel()
        W[:, j] = Wj.ravel()
        W[:, k] = 1.0 - Wi.ravel() - Wj.ravel()
        z = np.asarray(values_fn(W)).reshape(Wi.shape)
        grids.append(np.ma.masked_where(mask, z))

    if log_scale:
        assert all(bool(np.all(g.compressed() > 0)) for g in grids), \
            "log_scale=True requires every evaluated value to be strictly positive"

    if vmin is None:
        vmin = min(float(g.min()) for g in grids)
    if vmax is None:
        vmax = max(float(g.max()) for g in grids)
    if log_scale:
        # contourf's integer `levels` spaces bands linearly even under a log
        # norm, which collapses most of a wide-dynamic-range surface into a
        # single band; explicit log-spaced levels are needed instead.
        levels = np.geomspace(vmin, vmax, 21)
        color_kwargs = {"norm": LogNorm(vmin=vmin, vmax=vmax)}
    else:
        levels = 20
        color_kwargs = {"vmin": vmin, "vmax": vmax}

    fig, axes = plt.subplots(1, 4, figsize=(panel_size[0] * 3, panel_size[1]),
                              gridspec_kw={"width_ratios": [1, 1, 1, 0.06], "wspace": 0.35})
    panel_axes, cax = axes[:3], axes[3]
    mesh = None
    for ax, (i, j, k), z in zip(panel_axes, combos, grids):
        mesh = ax.contourf(Wi, Wj, z, levels=levels, cmap=cmap, **color_kwargs)
        _draw_simplex_boundary_2d(ax)
        if overlay_points is not None:
            pts = np.asarray(overlay_points)
            ax.scatter(pts[:, i], pts[:, j], s=10, color="white",
                       edgecolor="0.25", linewidth=0.4, alpha=0.85, zorder=4)
        ax.set_xlabel(labels[i], fontsize=11)
        ax.set_ylabel(labels[j], fontsize=11)
        ax.set_title(f"{labels[k]} implícito", fontsize=12, pad=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.grid(alpha=0.2)

    cbar = fig.colorbar(mesh, cax=cax)
    if log_scale:
        _style_log_colorbar(cbar, colorbar_label)
    elif colorbar_label:
        cbar.set_label(colorbar_label, fontsize=10)

    # tight_layout warns/misbehaves on axes with a fixed aspect ratio (the
    # panels are forced square above), so margins are set explicitly instead
    fig.subplots_adjust(bottom=0.12, top=0.84 if suptitle else 0.9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.97)
    if show:
        plt.show()
    return fig


def plot_surrogate_surface_3d(predict_fn: Any,
                               labels: Sequence[str] = ("x1", "x2", "x3"),
                               dropped_index: int = 2,
                               n_points: int = 45,
                               elev: float = 25,
                               azim: float = -50,
                               roll: float = 0,
                               title: Optional[str] = None,
                               ax=None,
                               figsize: Tuple[float, float] = (7, 6.5),
                               cmap: str = "viridis",
                               train_points: Optional[np.ndarray] = None,
                               train_values: Optional[np.ndarray] = None,
                               point_color: str = "white",
                               zlabel: Optional[str] = None,
                               show: bool = True):
    """
    Plot a fitted function's predicted surface over one 2D slice of the
    3-asset simplex (w_dropped = 1 - wi - wj held implicit), as a literal 3D
    surface, so the modeled landscape itself is visible rather than only a
    top-down color map. Real observations can be scattered directly on top,
    at their true (wi, wj, value) coordinates, as a visual fit check: points
    sitting on the surface indicate a good fit, points floating far above or
    below it indicate residual error the surrogate is smoothing over.

    predict_fn: callable, predict_fn(W) -> (n,) array of predicted values for
        an (n, 3) array of full simplex weights W (e.g.
        `lambda W: gpr.predict(W, return_std=False)` for a fitted SurrogateGPR).
    labels: the 3 asset names; the plot's x/y axes use the two not dropped.
    dropped_index: which of the 3 assets (0, 1, or 2) is treated as implicit
        (w_dropped = 1 - wi - wj); the other two become the plot's x/y axes.
    n_points: grid resolution per axis.
    elev, azim, roll: viewing angle in degrees, forwarded to Axes3D.view_init.
    title: optional plot title.
    ax: existing 3D axes to draw on; a new figure/axes are created when omitted.
    figsize: size of the created figure; ignored when ax is given.
    cmap: colormap for the surface.
    train_points: optional (n, 3) array of simplex weights to scatter on top
        of the surface.
    train_values: the true (observed, not predicted) value at each row of
        train_points; required whenever train_points is given.
    point_color: marker color for the overlaid points.
    zlabel: label for the z-axis; defaults to "predicted".
    show: whether to call plt.show() after building the figure.

    Returns: (fig, ax).
    """
    assert dropped_index in (0, 1, 2), f"dropped_index must be 0, 1 or 2 (current is {dropped_index})"
    i, j = [a for a in range(3) if a != dropped_index]
    k = dropped_index

    Wi, Wj, mask = _simplex_grid_2d(n_points)
    W = np.zeros((Wi.size, 3))
    W[:, i] = Wi.ravel()
    W[:, j] = Wj.ravel()
    W[:, k] = 1.0 - Wi.ravel() - Wj.ravel()
    Z = np.asarray(predict_fn(W)).reshape(Wi.shape).astype(float)
    Z[mask] = np.nan

    created_fig = ax is None
    if created_fig:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.plot_surface(Wi, Wj, Z, cmap=cmap, linewidth=0, antialiased=True, alpha=0.75, zorder=1)

    if train_points is not None:
        assert train_values is not None, "train_values is required whenever train_points is given"
        pts = np.asarray(train_points)
        ax.scatter(pts[:, i], pts[:, j], np.asarray(train_values),
                   color=point_color, edgecolor="0.25", linewidth=0.5, s=25,
                   depthshade=False, zorder=5)

    ax.set_xlabel(labels[i], labelpad=8, fontsize=11)
    ax.set_ylabel(labels[j], labelpad=8, fontsize=11)
    ax.set_zlabel(zlabel or "predicted", labelpad=6, fontsize=11)
    if title:
        ax.set_title(title, fontsize=13, pad=12)

    _style_3d_axes(ax, elev=elev, azim=azim, roll=roll, tick_step=None, box_aspect=None)

    if created_fig:
        fig.tight_layout()
        # this view angle puts the z-label on the right instead of the left
        # (as in the DoE/metric plots), so it is tight_layout's right margin
        # that underestimates the headroom Axes3D actually needs
        fig.subplots_adjust(right=0.88)
    if show:
        plt.show()
    return fig, ax


def plot_surrogate_comparison_3d(models: Dict[str, Any],
                                  labels: Sequence[str] = ("x1", "x2", "x3"),
                                  dropped_index: int = 2,
                                  train_points: Optional[np.ndarray] = None,
                                  train_values: Optional[np.ndarray] = None,
                                  n_points: int = 45,
                                  elev: float = 25,
                                  azim: float = -50,
                                  roll: float = 0,
                                  cmap: str = "viridis",
                                  panel_size: Tuple[float, float] = (6.5, 6),
                                  zlabel: Optional[str] = None,
                                  suptitle: Optional[str] = None,
                                  share_zlim: bool = True,
                                  show: bool = True):
    """
    Plot several fitted models' predicted surfaces side by side over the same
    2D simplex slice (e.g. candidate kernels fitted to the same DoE/target),
    each with the true training observations scattered on top, so their
    relative fit quality and shape assumptions are visually comparable.

    models: mapping from a display name (used as each panel's title) to a
        predict_fn(W) -> (n,) callable, forwarded to plot_surrogate_surface_3d.
    labels, dropped_index, n_points, elev, azim, roll, cmap: forwarded to
        plot_surrogate_surface_3d for every panel.
    train_points, train_values: shared across every panel — the same
        DoE/target every model in `models` was fitted on.
    panel_size: (width, height) of each individual panel.
    zlabel: label for every panel's z-axis.
    suptitle: optional figure-level title.
    share_zlim: if True, every panel uses the same z-axis limits, so
        differences in predicted amplitude across models are directly
        comparable rather than each panel auto-scaling to its own range.
    show: whether to call plt.show() after building the figure.

    Returns: fig.
    """
    names = list(models.keys())
    n = len(names)
    fig = plt.figure(figsize=(panel_size[0] * n, panel_size[1]))

    axes = []
    for idx, name in enumerate(names):
        ax = fig.add_subplot(1, n, idx + 1, projection="3d")
        plot_surrogate_surface_3d(models[name], labels=labels, dropped_index=dropped_index,
                                   n_points=n_points, elev=elev, azim=azim, roll=roll,
                                   title=name, ax=ax, cmap=cmap,
                                   train_points=train_points, train_values=train_values,
                                   zlabel=zlabel, show=False)
        axes.append(ax)

    if share_zlim:
        zlo = min(ax.get_zlim()[0] for ax in axes)
        zhi = max(ax.get_zlim()[1] for ax in axes)
        for ax in axes:
            ax.set_zlim(zlo, zhi)

    fig.tight_layout(w_pad=4.0)
    fig.subplots_adjust(left=0.06, right=0.96, top=0.86 if suptitle else 0.94)
    if suptitle:
        fig.suptitle(suptitle, fontsize=14, y=0.98)
    if show:
        plt.show()
    return fig


def _simplex_triangulated_mesh(n: int):
    """
    Triangular lattice covering the 2-simplex {(u, v): u >= 0, v >= 0, u+v <= 1},
    built directly from barycentric grid points rather than a rectangular grid
    with the outside masked off. Every lattice point already lies exactly on or
    inside the simplex, and the 3 boundary edges fall exactly on lattice lines,
    so triangulating it gives a perfectly straight hypotenuse — unlike masking a
    rectangular grid, which leaves a jagged, staircase-shaped edge along the cut.

    n: lattice resolution per edge (the mesh has (n+1)(n+2)/2 vertices).

    Returns (u, v, triangles): 1D coordinate arrays and an (n_tris, 3) array of
    vertex-index triplets, suitable for go.Mesh3d's i/j/k.
    """
    idx = {}
    pts = []
    for a in range(n + 1):
        for b in range(n + 1 - a):
            idx[(a, b)] = len(pts)
            pts.append((a, b))
    pts = np.asarray(pts, dtype=float) / n

    tris = []
    for a in range(n):
        for b in range(n - a):
            tris.append((idx[(a, b)], idx[(a + 1, b)], idx[(a, b + 1)]))
            if (a + 1, b + 1) in idx:
                tris.append((idx[(a + 1, b)], idx[(a + 1, b + 1)], idx[(a, b + 1)]))

    return pts[:, 0], pts[:, 1], np.asarray(tris)


def _surrogate_surface_mesh(predict_fn: Any, dropped_index: int, n_points: int):
    """Shared mesh-building step for the interactive (Plotly) surface plots; see plot_surrogate_surface_3d."""
    assert dropped_index in (0, 1, 2), f"dropped_index must be 0, 1 or 2 (current is {dropped_index})"
    i, j = [a for a in range(3) if a != dropped_index]
    k = dropped_index
    Wi, Wj, tris = _simplex_triangulated_mesh(n_points)
    W = np.zeros((Wi.size, 3))
    W[:, i] = Wi
    W[:, j] = Wj
    W[:, k] = 1.0 - Wi - Wj
    Z = np.asarray(predict_fn(W)).astype(float)
    return i, j, Wi, Wj, Z, tris


def plot_surrogate_surface_3d_interactive(predict_fn: Any,
                                           labels: Sequence[str] = ("x1", "x2", "x3"),
                                           dropped_index: int = 2,
                                           n_points: int = 45,
                                           title: Optional[str] = None,
                                           train_points: Optional[np.ndarray] = None,
                                           train_values: Optional[np.ndarray] = None,
                                           zlabel: Optional[str] = None,
                                           colorscale: str = "Viridis",
                                           point_color: str = "white",
                                           width: int = 900,
                                           height: int = 650):
    """
    Interactive, mouse-rotatable counterpart to plot_surrogate_surface_3d, built
    with Plotly instead of matplotlib. Two things matplotlib's software 3D
    renderer cannot do reliably that this gets for free from Plotly's WebGL
    engine: free rotation/zoom by dragging, without re-specifying elev/azim/roll,
    and correct per-pixel depth occlusion between the surface and the overlaid
    points — a training point the surface passes through renders as genuinely
    half-hidden behind it, not simply drawn entirely in front of or behind.

    predict_fn: callable, predict_fn(W) -> (n,) array of predicted values for
        an (n, 3) array of full simplex weights W.
    labels: the 3 asset names; the plot's x/y axes use the two not dropped.
    dropped_index: which of the 3 assets (0, 1, or 2) is treated as implicit
        (w_dropped = 1 - wi - wj); the other two become the plot's x/y axes.
    n_points: grid resolution per axis.
    title: optional figure title.
    train_points: optional (n, 3) array of simplex weights to scatter on top
        of the surface.
    train_values: the true (observed, not predicted) value at each row of
        train_points; required whenever train_points is given.
    zlabel: label for the z-axis; defaults to "predicted".
    colorscale: Plotly colorscale name for the surface.
    point_color: marker color for the overlaid points.
    width, height: figure size in pixels.

    Returns: a plotly.graph_objects.Figure. Call .show() to render it in the
        notebook (drag to rotate, scroll/pinch to zoom).
    """
    i, j, Wi, Wj, Z, tris = _surrogate_surface_mesh(predict_fn, dropped_index, n_points)

    fig = go.Figure()
    fig.add_trace(go.Mesh3d(x=Wi, y=Wj, z=Z, i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                             intensity=Z, colorscale=colorscale, opacity=0.92,
                             showscale=True,
                             colorbar=dict(title=dict(text=zlabel or "predicted"), len=0.7)))

    if train_points is not None:
        assert train_values is not None, "train_values is required whenever train_points is given"
        pts = np.asarray(train_points)
        fig.add_trace(go.Scatter3d(
            x=pts[:, i], y=pts[:, j], z=np.asarray(train_values),
            mode="markers",
            marker=dict(size=4, color=point_color, line=dict(color="black", width=1)),
            name="observado",
        ))

    fig.update_layout(
        title=title,
        width=width, height=height,
        scene=dict(
            xaxis=dict(title=dict(text=labels[i])),
            yaxis=dict(title=dict(text=labels[j])),
            zaxis=dict(title=dict(text=zlabel or "predicted")),
            aspectmode="cube",
        ),
        margin=dict(l=80, r=10, b=40, t=50 if title else 10),
        showlegend=False,
    )
    return fig


def plot_surrogate_comparison_3d_interactive(models: Dict[str, Any],
                                              labels: Sequence[str] = ("x1", "x2", "x3"),
                                              dropped_index: int = 2,
                                              train_points: Optional[np.ndarray] = None,
                                              train_values: Optional[np.ndarray] = None,
                                              n_points: int = 45,
                                              zlabel: Optional[str] = None,
                                              colorscale: str = "Viridis",
                                              point_color: str = "white",
                                              suptitle: Optional[str] = None,
                                              share_zlim: bool = True,
                                              panel_width: int = 650,
                                              height: int = 620):
    """
    Interactive, mouse-rotatable counterpart to plot_surrogate_comparison_3d:
    several fitted models' predicted surfaces side by side, each with the true
    training observations scattered on top with correct depth occlusion. Every
    panel gets its own independently rotatable 3D scene (dragging one does not
    rotate the others).

    models: mapping from a display name (used as each panel's title) to a
        predict_fn(W) -> (n,) callable.
    labels, dropped_index, n_points, colorscale, point_color: forwarded to
        every panel; see plot_surrogate_surface_3d_interactive.
    train_points, train_values: shared across every panel — the same
        DoE/target every model in `models` was fitted on.
    zlabel: label for every panel's z-axis.
    suptitle: optional figure-level title.
    share_zlim: if True, every panel uses the same z-axis range, so
        differences in predicted amplitude across models are directly
        comparable rather than each panel auto-scaling to its own range.
    panel_width: width in pixels of each individual panel.
    height: figure height in pixels.

    Returns: a plotly.graph_objects.Figure. Call .show() to render it in the
        notebook.
    """
    names = list(models.keys())
    n = len(names)

    fig = make_subplots(rows=1, cols=n, specs=[[{"type": "scene"}] * n],
                         subplot_titles=names, horizontal_spacing=0.06)

    all_i = all_j = None
    all_z = []
    for idx, name in enumerate(names):
        i, j, Wi, Wj, Z, tris = _surrogate_surface_mesh(models[name], dropped_index, n_points)
        all_i, all_j = i, j
        all_z.append(Z)
        fig.add_trace(go.Mesh3d(x=Wi, y=Wj, z=Z, i=tris[:, 0], j=tris[:, 1], k=tris[:, 2],
                                 intensity=Z, colorscale=colorscale, opacity=0.92,
                                 showscale=(idx == 0),
                                 colorbar=dict(title=dict(text=zlabel or "predicted"), len=0.7,
                                               x=1.0 / n * (idx + 1) - 0.02) if idx == 0 else None),
                      row=1, col=idx + 1)
        if train_points is not None:
            assert train_values is not None, "train_values is required whenever train_points is given"
            pts = np.asarray(train_points)
            fig.add_trace(go.Scatter3d(
                x=pts[:, i], y=pts[:, j], z=np.asarray(train_values),
                mode="markers", marker=dict(size=3, color=point_color, line=dict(color="black", width=0.8)),
                showlegend=False),
                row=1, col=idx + 1)

    zaxis_range = None
    if share_zlim:
        zlo = min(float(np.nanmin(z)) for z in all_z)
        zhi = max(float(np.nanmax(z)) for z in all_z)
        zaxis_range = [zlo, zhi]

    # axis titles are only kept on the leftmost panel: every panel shares the
    # same 2 dropped-asset axes and z quantity (already named by the colorbar),
    # so repeating the same 3 titles n times only crowds each panel's margin
    # without adding information — tick values (the actual per-panel content)
    # stay on every panel regardless.
    def _scene(first: bool) -> dict:
        return dict(
            xaxis=dict(title=dict(text=labels[all_i] if first else "")),
            yaxis=dict(title=dict(text=labels[all_j] if first else "")),
            zaxis=dict(title=dict(text=(zlabel or "predicted") if first else ""),
                       range=zaxis_range),
            aspectmode="cube",
        )
    layout_update = {("scene" if idx == 0 else f"scene{idx + 1}"): _scene(idx == 0) for idx in range(n)}
    fig.update_layout(
        title=suptitle,
        width=panel_width * n, height=height,
        margin=dict(l=90, r=10, b=40, t=100 if suptitle else 60),
        **layout_update,
    )
    return fig

