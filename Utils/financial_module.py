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
    
    def __repr__(self):
        return f"<Object: Dataclass StockPortfolio> (n_assets={self.num_assets} tickers={self.stocks})"





def _expected_portfolio_return(w: np.ndarray, R: np.ndarray, mu: Optional[np.ndarray], n_assets: int) -> float:
    """
    Computes w^T mu, the portfolio's expected return </p>
    mu (np.ndarray, optional): expected return vector (length n_assets). When omitted, the historical
    sample mean of R is used as a default estimator. Pass an externally computed vector (e.g. CAPM-implied
    returns, or an arbitrary view held by the investor) to keep the expected return decoupled from the
    historical sample used to estimate risk.
    """
    if mu is None:
        mu = R.mean(axis=0)
    else:
        mu = np.asarray(mu, dtype=float)
        assert mu.shape == (n_assets,), f"mu must have shape ({n_assets},) (current is {mu.shape})"
    return float(w @ mu)


class PortfolioVariance:
    """
    PortfolioVariance Class is a Callable Object implementing Markowitz's mean-variance model.
    Any instance of PortfolioVariance deals only with a predefined number of assets to avoid mistaken
    comparison between portfolios of different characteristics.
    """
    def __init__(self, n_assets: int):
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        self.n_assets = n_assets

    def __repr__(self):
        return f"<Object: Callable PortfolioVariance> (n_assets={self.n_assets})"


    def getCovariance(self, R: np.ndarray) -> np.ndarray:
        """
        Returns the sample covariance matrix of asset returns </p>
        R (np.ndarray): asset returns matrix of shape (T, n_assets), one column per asset
        """
        assert R.ndim == 2, f"R array must be 2-Dimensional (current is {R.ndim}D)"
        assert R.shape[1] == self.n_assets, f"R must have {self.n_assets} columns (current is {R.shape[1]})"
        return np.cov(R, rowvar=False)

    def __call__(self, w: np.ndarray, R: np.ndarray, mu: Optional[np.ndarray] = None) -> tuple[float, float]:
        """
        Computes the Markowitz mean-variance objective function for portfolio weights w </p>
        w (np.ndarray): 1-Dimensional array of portfolio weights (length n_assets) </p>
        R (np.ndarray): asset returns matrix of shape (T, n_assets), used to estimate risk (and,
            when mu is not provided, also used as the historical estimator of expected return) </p>
        mu (np.ndarray, optional): expected return vector (length n_assets), estimated externally
            (e.g. via CAPM or an investor's own view). Defaults to the historical mean of R </p>
        Returns (mu_p, sigma2_p): the portfolio's expected return and variance
        """
        assert w.ndim == 1, f"w array must be 1-Dimensional (current is {w.ndim}D)"
        assert w.size == self.n_assets, f"w must have {self.n_assets} entries (current is {w.size})"
        cov = self.getCovariance(R)
        mu_p = _expected_portfolio_return(w, R, mu, self.n_assets)
        sigma2_p = float(w @ cov @ w)
        return mu_p, sigma2_p


class SharpeIndex(PortfolioVariance):
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

    def _PortfolioVariance(self, w: np.ndarray, R: np.ndarray, mu: Optional[np.ndarray] = None):
        return super().__call__(w, R, mu)

    def __call__(self, w: np.ndarray, R: np.ndarray, mu: Optional[np.ndarray] = None) -> float:
        """
        Computes the Sharpe Index (Sharpe Ratio) objective function for a portfolio return series </p>
        w (np.ndarray): 1-Dimensional array of portfolio weights (length n_assets) </p>
        R (np.ndarray): asset returns matrix of shape (T, n_assets), used to estimate risk (and,
            when mu is not provided, also used as the historical estimator of expected return) </p>
        mu (np.ndarray, optional): expected return vector (length n_assets), estimated externally
            (e.g. via CAPM or an investor's own view). Defaults to the historical mean of R </p>
        Returns the portfolios Sharpe Index
        """
        mu_p, sigma2_p = self._PortfolioVariance(w, R, mu)
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
        Computes the Value at Risk (VaR) for a portfolio return series </p>
        ret (np.ndarray): 1-Dimensional portfolio return series (e.g. historical or
            bootstrapped scenarios) used to estimate the alpha-quantile tail </p>
        Returns, when type="non-parametric": (var, ret_sorted), where var is the
            empirical alpha-quantile of ret (an actual observed value, since it is
            computed with method="higher" rather than interpolated) and ret_sorted
            is ret sorted ascending, reused by CVaR to avoid re-sorting </p>
        Returns, when type="parametric": (var, mu, sigma), where var assumes ret
            is normally distributed (var = mu + z_alpha * sigma) and mu, sigma are
            the sample mean and standard deviation of ret, reused by CVaR and
            plotDistribution to avoid recomputation </p>
        var is expressed as a return threshold, not a loss magnitude: for typical
        confidence levels it is negative, and losses worse than it are expected
        to occur with probability alpha
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

        cvar, var = self(ret)
        if self.type == "parametric":
            _, mu, sigma = self._VaR(ret)
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
    
    def __call__(self, ret: np.ndarray) -> tuple[float, float]:
        """
        Computes the Conditional Value at Risk (CVaR, a.k.a. Expected Shortfall)
        for a portfolio return series </p>
        ret (np.ndarray): 1-Dimensional portfolio return series, forwarded to
            VaR to obtain the alpha-quantile threshold </p>
        Returns (cvar, var), regardless of type </p>
        When type="non-parametric": cvar is the average of every observation
            in ret at or below var (the tail beyond the VaR threshold), and
            var is the empirical alpha-quantile computed by VaR </p>
        When type="parametric": both cvar and var assume ret is normally
            distributed; var = mu + z_alpha * sigma (as computed by VaR) and
            cvar is the closed-form tail expectation under that same normal
            model </p>
        cvar, like var, is expressed as a return rather than a loss magnitude,
        and is typically more negative than var since it averages over the
        tail instead of only marking its boundary
        """
        if self.type == "non-parametric":
            var, ret_sorted = self._VaR(ret)
            tail = ret_sorted[ret_sorted <= var]
            return tail.mean(), var
        elif self.type == "parametric":
            var, mu, sigma = self._VaR(ret)
            z_alpha = norm.ppf(self.alpha)
            phi_z = norm.pdf(z_alpha)  # densidade da normal padrão em z_alpha
            cvar = mu - sigma * phi_z / self.alpha
            return cvar, var
        else:
            message = "no valid type was found therefore no calculation war returned"
            raise ValueError(message)
        


class STARRIndex(CVaR):
    """
    STARRIndex Class is a Callable Object implemented so that any instance of STARRIndex deals only with
    a predefined set of portfolio cases to avoid mistaken comparison between portfolios of different characteristics.
    In this sense, STARRIndex predefines the risk-free rate, confidence level, number of assets and type of calculation.
    """
    def __init__(self, n_assets: int, rf: float, alpha: float,
                 type: Literal["non-parametric", "parametric"] = "non-parametric"):
        assert rf > 0, f"rf must be > 0 (current is {rf})"
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        super().__init__(n_assets=n_assets, alpha=alpha, type=type)
        self.n_assets = n_assets
        self.rf = rf

    def __repr__(self):
        return f"<Object: Callable STARRIndex> (n_assets={self.n_assets}; rf={self.rf}; alpha={self.alpha}; type={self.type})"

    def _CVaR(self, ret: np.ndarray):
        return super().__call__(ret)

    def __call__(self, ret: np.ndarray, mu: Optional[float] = None) -> tuple[float, float, float]:
        """
        Computes the STARR Ratio objective function for a portfolio return series </p>
        ret (np.ndarray): 1-Dimensional portfolio return series </p>
        mu (float, optional): expected portfolio return, estimated externally (e.g. via CAPM
            or an investor's own view on this specific portfolio). Defaults to the historical
            mean of ret </p>
        Returns (starr, cvar, var): the portfolio's STARR Index together with
            the CVaR and VaR it was computed from (type="non-parametric" or
            "parametric", per how this instance was configured) </p>
        """
        mu_p = float(ret.mean()) if mu is None else float(mu)
        cvar_p, var_p = self._CVaR(ret)
        # CVaR here is the average tail *return* (typically negative); STARR's denominator
        # is a loss magnitude, so it is the sign-flipped, positive counterpart of that value.
        starr = (mu_p - self.rf) / -cvar_p
        return starr, cvar_p, var_p




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





class ReturnsEstimator:
    "Return estimator namespace containing..."

    class Average:
        ...

    class CAPM:
        ...
