from pathlib import Path

import matplotlib.pyplot as plt

from Utils.extraction_module import StockDataWarehouse


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
