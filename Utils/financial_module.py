import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Literal, Optional, Tuple, Union
from dataclasses import dataclass, field
from scipy.stats import norm, gaussian_kde


@dataclass
class StockPortfolio:
    """A class to represent a stock portfolio </p>Attributes:</p>
    stocks (Tuple[str]): A tuple of stock ticker symbols in the portfolio.</p>
    composition (Tuple[float]): A tuple of corresponding weights for each stock in the portfolio.
    """
    stocks: Tuple[str] = field(default_factory=tuple) #tickers should not be mutable, therefore justifies using tuple instead of list
    composition: Tuple[float] = field(default_factory=tuple) #weights should not be mutable, therefore justifies using tuple instead of list

    def __post_init__(self) -> None:
        if self.composition and not math.isclose(sum(self.composition), 1.0, abs_tol=1e-8):
            raise ValueError(f"Composition weights must sum to 1.0 (current sum is {sum(self.composition)}).")
        if self.composition and len(self.composition) != len(self.stocks):
            raise ValueError("composition must have the same length as stocks.")

    def alter_composition(self, new_composition: Tuple[float]) -> None:
        """
        Alter the composition of the portfolio.</p>Parameters:</p>
        new_composition (Tuple[float]): A tuple of new weights for each stock in the portfolio.
        """
        if len(new_composition) != len(self.stocks):
            raise ValueError("New composition must have the same length as the number of stocks.")
        self.composition = new_composition
    
    def num_assets(self):
        return len(self.stocks)



class MeanVariance:
    """
    MeanVariance Class is a Callable Object implementing Markowitz's mean-variance model.
    Any instance of MeanVariance deals only with a predefined number of assets to avoid mistaken
    comparison between portfolios of different characteristics.
    """
    def __init__(self, n_assets: int):
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        self.n_assets = n_assets

    def __repr__(self):
        return f"<Object: Callable MeanVariance> (n_assets={self.n_assets})"


    def getCovariance(self, R: np.ndarray) -> np.ndarray:
        """
        Returns the sample covariance matrix of asset returns </p>
        R (np.ndarray): asset returns matrix of shape (T, n_assets), one column per asset
        """
        assert R.ndim == 2, f"R array must be 2-Dimensional (current is {R.ndim}D)"
        assert R.shape[1] == self.n_assets, f"R must have {self.n_assets} columns (current is {R.shape[1]})"
        return np.cov(R, rowvar=False)

    def __call__(self, w: np.ndarray, R: np.ndarray) -> tuple[float, float]:
        """
        Computes the Markowitz mean-variance objective function for portfolio weights w </p>
        w (np.ndarray): 1-Dimensional array of portfolio weights (length n_assets) </p>
        R (np.ndarray): asset returns matrix of shape (T, n_assets) </p>
        Returns (mu_p, sigma2_p): the portfolio's expected return and variance
        """
        assert w.ndim == 1, f"w array must be 1-Dimensional (current is {w.ndim}D)"
        assert w.size == self.n_assets, f"w must have {self.n_assets} entries (current is {w.size})"
        cov = self.getCovariance(R)
        mu = R.mean(axis=0)
        mu_p = float(w @ mu)
        sigma2_p = float(w @ cov @ w)
        return mu_p, sigma2_p


class SharpeIndex(MeanVariance):
    """
    SharpeIndex Class is a Callable Object implemented so that any instance of SharpeIndex deals only with
    a predefined set of portfolio cases to avoid mistaken comparison between portfolios of different characteristics.
    In this sense, SharpeIndex predefines the risk-free rate and number of assets.
    """
    def __init__(self, n_assets: int, rf: float):
        assert rf > 0, f"rf must be > 0 (current is {rf})"
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        super().__init__(n_assets=n_assets)
        self.n_assets = n_assets
        self.rf = rf

    def __repr__(self):
        return f"<Object: Callable SharpeIndex> (n_assets={self.n_assets}; rf={self.rf})"

    def _MeanVariance(self, w:np.ndarray, R: np.ndarray):
        return super().__call__(w, R)

    def __call__(self, w:np.ndarray, R: np.ndarray) -> float:
        """
        Computes the Sharpe Index (Sharpe Ratio) objective function for a portfolio return series </p>
        w (np.ndarray): 1-Dimensional array of portfolio weights (length n_assets) </p>
        R (np.ndarray): asset returns matrix of shape (T, n_assets) </p>
        Returns the portfolios Sharpe Index
        """
        mu_p, sigma2_p = self._MeanVariance(w, R)
        return (mu_p - self.rf)/np.sqrt(sigma2_p)



class VaR:
    """
    VaR Class is a Callable Object implemented so that any instance of VaR deals only with 
    a predefined set of portfolio cases to avoid mistaken comparison between portfolios of different characteristics.
    In this sense, VaR predefines alpha, number of assets and type of calculation.
    """
    def __init__(self, n_assets: int, 
                 alpha:float, 
                 type:Literal["non-parametric", "parametric"]="non-parametric"):
        assert 0<=alpha<=1, f"alpha must be between 0 and 1 (current is {alpha})"
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        assert type in ["non-parametric", "parametric"], "type must be one of Literal['non-parametric', 'parametric', 'linearized']"
        self.n_assets = n_assets
        self.alpha = alpha
        self.type = type

    def __repr__(self):
        return f"<Object: Callable VaR> (n_assets={self.n_assets}; alpha={self.alpha}; type={self.type})"
    
    def orderDistibution(self, ret: np.ndarray):
        return np.sort(ret)
    
    def plotDistribution(self, ret: np.ndarray, bins: int = 50,
                          capital: Optional[float] = None, title: Optional[str] = None) -> plt.Figure:
        """
        Plots the returns distribution indicating the VaR threshold </p>
        ret (np.ndarray): 1-Dimensional portfolio return series </p>
        bins (int): number of histogram bins </p>
        capital (float, optional): symbolic capital used to add a secondary monetary axis
            (top x-axis) and monetary amounts in the VaR label; omit to keep everything in returns </p>
        title (str, optional): overrides the default axes title (e.g. to identify the portfolio) </p>
        Returns the matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ret, bins=bins, density=True, color="steelblue", alpha=0.4, edgecolor="white", label="Retornos")

        x = np.linspace(ret.min(), ret.max(), 300)
        ax.plot(x, gaussian_kde(ret)(x), color="steelblue", linewidth=2, label="Densidade (KDE)")

        if self.type == "non-parametric":
            var, _ = self(ret)
        else:
            var, mu, sigma = self(ret)
            ax.plot(x, norm.pdf(x, mu, sigma), color="black", linewidth=1.5, linestyle="--", label="Normal ajustada")

        var_label = f"VaR ({self.alpha:.0%}) = {var:.2%}"
        if capital is not None:
            var_label += f" (R$ {var * capital:,.2f})"

        ax.axvspan(ax.get_xlim()[0], var, color="red", alpha=0.15, label="Cauda (≤ VaR)")
        ax.axvline(var, color="red", linestyle="--", linewidth=2, label=var_label)

        ax.set_title(title or f"Distribuição de retornos e VaR ({self.type})")
        ax.set_xlabel("Retorno")
        ax.set_ylabel("Densidade")
        if capital is not None:
            money_axis = ax.secondary_xaxis("top", functions=(lambda x: x * capital, lambda x: x / capital))
            money_axis.set_xlabel(f"Resultado sobre capital de R$ {capital:,.2f}")
        ax.legend(loc="upper left", fontsize="small")
        fig.tight_layout()
        return fig

    def __call__(self, ret: np.ndarray) -> tuple[float, np.ndarray] | tuple[float, float, float]:
        """
        Returns VaR and ordered ret array if method is non-parametric </p>
        Returns VaR, mu and sigma if method is parametric
        
        """
        assert ret.ndim == 1, f"ret array must be 1-Dimensional (current is {ret.ndim}D)"
        assert ret.size > 0, f"ret array must contain more than 0 entries (current is {ret.size})"
        if self.type == "non-parametric":
            ret_sorted = self.orderDistibution(ret)
            return np.percentile(ret, self.alpha*100, method="higher"), ret_sorted
        elif self.type == "parametric":
            mu = ret.mean()
            sigma = ret.std(ddof=1)
            z_alpha = norm.ppf(self.alpha)
            var_return = mu + z_alpha * sigma
            return var_return, mu, sigma
        else:
            message = "no valid type was found therefore no calculation war returned"
            raise ValueError(message)


class CVaR(VaR):
    """
    CVaR Class is a Callable Object implemented so that any instance of CVaR deals only with 
    a predefined set of portfolio cases to avoid mistaken comparison between portfolios of different characteristics.
    In this sense, CVaR predefines alpha, number of assets and type of calculation.
    """
    def __init__(self, n_assets: int, 
                 alpha:float, 
                 type: Literal["non-parametric", "parametric"]="non-parametric"):
        assert 0<=alpha<=1, f"alpha must be between 0 and 1 (current is {alpha})"
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        assert type in ["non-parametric", "parametric", "linearized"], "type must be one of Literal['non-parametric', 'parametric', 'linearized']"
        super().__init__(n_assets, alpha, type)
        self.n_assets = n_assets
        self.alpha = alpha
        self.type = type
    
    def _VaR(self, ret: np.ndarray):
        return super().__call__(ret)

    def __repr__(self):
        return f"<Object: Callable CVaR> (n_assets={self.n_assets}; alpha={self.alpha}; type={self.type})"
    
    def plotDistribution(self, ret: np.ndarray, bins: int = 50,
                          capital: Optional[float] = None, title: Optional[str] = None) -> plt.Figure:
        """
        Plots the returns distribution indicating the VaR and CVaR thresholds </p>
        ret (np.ndarray): 1-Dimensional portfolio return series </p>
        bins (int): number of histogram bins </p>
        capital (float, optional): symbolic capital used to add a secondary monetary axis
            (top x-axis) and monetary amounts in the VaR/CVaR labels; omit to keep everything
            in returns </p>
        title (str, optional): overrides the default axes title (e.g. to identify the portfolio) </p>
        Returns the matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(ret, bins=bins, density=True, color="steelblue", alpha=0.4, edgecolor="white", label="Retornos")

        x = np.linspace(ret.min(), ret.max(), 300)
        ax.plot(x, gaussian_kde(ret)(x), color="steelblue", linewidth=2, label="Densidade (KDE)")

        if self.type == "non-parametric":
            cvar, var, _ = self(ret)
        else:
            var, mu, sigma = self._VaR(ret)
            cvar = self(ret)
            ax.plot(x, norm.pdf(x, mu, sigma), color="black", linewidth=1.5, linestyle="--", label="Normal ajustada")

        var_label = f"VaR ({self.alpha:.0%}) = {var:.2%}"
        cvar_label = f"CVaR ({self.alpha:.0%}) = {cvar:.2%}"
        if capital is not None:
            var_label += f" (R$ {var * capital:,.2f})"
            cvar_label += f" (R$ {cvar * capital:,.2f})"

        ax.axvspan(ax.get_xlim()[0], var, color="red", alpha=0.15, label="Cauda (≤ VaR)")
        ax.axvline(var, color="red", linestyle="--", linewidth=2, label=var_label)
        ax.axvline(cvar, color="darkred", linestyle="-", linewidth=2, label=cvar_label)

        ax.set_title(title or f"Distribuição de retornos, VaR e CVaR ({self.type})")
        ax.set_xlabel("Retorno")
        ax.set_ylabel("Densidade")
        if capital is not None:
            money_axis = ax.secondary_xaxis("top", functions=(lambda x: x * capital, lambda x: x / capital))
            money_axis.set_xlabel(f"Resultado sobre capital de R$ {capital:,.2f}")
        ax.legend(loc="upper left", fontsize="small")
        fig.tight_layout()
        return fig
    
    def __call__(self, ret: np.ndarray) -> tuple[float, float, np.ndarray]:
        """Returns CVaR, VaR and ordered ret array"""
        if self.type == "non-parametric":
            var, ret_sorted = self._VaR(ret)
            tail = ret_sorted[ret_sorted <= var]
            return  tail.mean(), var, ret_sorted
        elif self.type == "parametric":
            var, mu, sigma = self._VaR(ret)
            z_alpha = norm.ppf(self.alpha)
            phi_z = norm.pdf(z_alpha)  # densidade da normal padrão em z_alpha
            return mu - sigma * phi_z / self.alpha
        else:
            message = "no valid type was found therefore no calculation war returned"
            raise ValueError(message)
        


class ReturnSampler:
    """
    Samples historical return series from a price panel.

    Receives a (T, n_assets) price matrix (e.g. StockDataWarehouse.price_panel),
    converts it to log-returns internally, draws horizon windows shared across
    all assets, and returns a (n_windows, n_assets) matrix of simple returns
    over the horizon.

    Sampling mode is controlled by n_samples:

    n_samples=None -> historical, non-overlapping windows.
        Window starts at 0, h, 2h, ... covering the full range once, with no
        overlap and no reused days. Deterministic, seed has no effect, yields
        exactly T // h samples. Use for pure historical VaR/CVaR when the
        series is long enough.

    n_samples=K (int) -> bootstrap.
        K window starts drawn with replacement anywhere in [0, T - h]; windows
        may overlap or repeat. Number of samples is decoupled from series
        length and the draw is fixed by seed. Use when a large horizon leaves
        T // h too small to estimate the tail.

    Parameters:
    horizon (int): number of periods aggregated into each sampled return.
    n_samples (int | None): None for non-overlapping historical windows,
        int for bootstrap with that many samples.
    seed (int | None): fixes the bootstrap draw; ignored when n_samples is None.
    """

    def __init__(self,
                 horizon: int = 1,
                 n_samples: Optional[int] = None,
                 seed: Optional[int] = 0):
        assert horizon >= 1, f"horizon must be >= 1 (current is {horizon})"
        if n_samples is not None:
            assert n_samples > 0, f"n_samples must be > 0 (current is {n_samples})"
        self.horizon = horizon
        self.n_samples = n_samples   # None -> non-overlapping windows; int -> bootstrap
        self.seed = seed

    def __repr__(self):
        return (f"<Object: Callable ReturnSampler> (horizon={self.horizon}; "
                f"n_samples={self.n_samples}; seed={self.seed})")

    @staticmethod
    def _to_log_returns(prices: np.ndarray) -> np.ndarray:
        """Convert a price matrix (or series) to log-returns along the time axis."""
        prices = np.asarray(prices, dtype=float)
        assert prices.ndim in (1, 2), \
            f"prices must be 1-D or 2-D (current is {prices.ndim}D)"
        assert np.all(prices > 0), "prices must be strictly positive"
        return np.diff(np.log(prices), axis=0)

    def _window_starts(self, T: int) -> np.ndarray:
        """Window starting indices, identical for every asset in the matrix."""
        h = self.horizon
        assert T >= h, f"return series too short ({T}) for horizon {h}"
        if self.n_samples is None:
            return np.arange(T // h) * h
        rng = np.random.default_rng(self.seed)
        return rng.integers(0, T - h + 1, size=self.n_samples)

    def _aggregate(self, log_returns: np.ndarray, starts: np.ndarray) -> np.ndarray:
        """Sum log-returns over each horizon window (per asset), convert to simple returns."""
        h = self.horizon
        idx = starts[:, None] + np.arange(h)[None, :]   # (n_windows, h)
        horizon_log = log_returns[idx].sum(axis=1)      # collapse the horizon axis
        return np.expm1(horizon_log)                    # back to simple returns

    def __call__(self, prices: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Sample a return matrix from a (T, n_assets) price panel.
        A 1-D price series is accepted and returned as a 1-D sampled series.
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices.to_numpy()
        elif isinstance(prices, pd.Series):
            prices = prices.to_numpy()
        log_returns = self._to_log_returns(prices)
        starts = self._window_starts(log_returns.shape[0])
        return self._aggregate(log_returns, starts)