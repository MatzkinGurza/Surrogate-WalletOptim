import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Literal, Optional, Tuple, Union
from dataclasses import dataclass, field

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel, DotProduct, RBF, RationalQuadratic
from sklearn.gaussian_process.kernels import (
    Matern,           # o componente de resíduo local suave
    ConstantKernel,   # veículo da amplitude σ_f (a que multiplica o Matérn)
    WhiteKernel,      # ruído de estimativa
    DotProduct,       # base para tendência polinomial
    RBF,              # kernel simples infinitamente diferenciável - funções exageradamente lisas
    RationalQuadratic # mistura de RBFs em multiplas escalas
)
from sklearn.gaussian_process.kernels import Kernel
from scipy.linalg import helmert

import warnings
from typing import Optional

import numpy as np
from scipy.linalg import helmert
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel



class SurrogateGPR:
    """
    Gaussian Process surrogate for a portfolio measure over the asset simplex.
 
    Portfolios live on the simplex (sum(w) = 1), so one input direction carries
    no variance and the raw covariance matrix is singular. Inputs are projected
    onto the Helmert subspace before fitting, which removes that degenerate
    direction without distorting distances between portfolios. Helmert (a linear
    isometry) is used rather than a log-ratio transform because risk measures are
    additive in the weights, not multiplicative.
    """
 
    def __init__(self, kernel: Kernel,
                 n_assets: int,
                 n_restarts: int,
                 sim_obj: Optional[str],
                 random_state: int = 0,
                 alpha_jitter: float = 5e-7):
        """
        kernel: initial prior (a scikit-learn Kernel), used as the starting
            point for hyperparameter optimization.
        n_assets: number of assets; sets the simplex dimension and the Helmert
            basis shape.
        n_restarts: restarts of the hyperparameter optimizer; higher values
            reduce the chance of settling in a poor local optimum of the ARD
            marginal likelihood.
        sim_obj: label of the objective being modelled (e.g. 'cvar'); metadata
            only, shown in repr.
        random_state: seed for reproducible restarts and posterior sampling.
        alpha_jitter: fixed diagonal added to the covariance for numerical
            safety against a non-invertible matrix when points are very close;
            not a noise model.
        """
        self.kernel = kernel                # initial prior
        self.kernel_ = None                 # fitted kernel, filled after fit
        self.n_restarts = n_restarts
        self.sim_obj = sim_obj
        self.random_state = random_state
        self.n_assets = n_assets
        self.jitter = alpha_jitter          # guards against a singular matrix when points nearly coincide
 
        self._U = helmert(n_assets)         # Helmert basis for the isometric projection (avoids degeneracy)
        # Helmert is used because risk measures are additive, not multiplicative, in the weights
 
        self.gp_ = None                     # holds the regressor after fit
        self.warm = False
 
    def _prepare(self, W):
        """
        Projects simplex portfolios onto the Helmert subspace.
        W: portfolio weights (K x n_assets), each row summing to 1.
        Returns the isometric projection X (K x n_assets-1), removing the
        degenerate direction sum(w)=1 while preserving distances.
        """
        try:
            W = np.asarray(W, dtype=float)
            return W @ self._U.T
        except Exception as e:
            raise ValueError(f"_prepare exited with exception: {e} for values W={W}, self._U={self._U}")
 
    def _length_scales(self):
        """
        Collects every length_scale hyperparameter from the fitted kernel.
        Handles all three cases: isotropic (scalar length_scale), single ARD
        (one vector), and multiple ARD terms (several vectors).
        Returns a list of 1D arrays, one per length_scale term found, each
        broadcast to length n_assets-1 so isotropic terms are directly
        comparable to ARD ones. Raises RuntimeError if none is found.
        """
        n_dims = self.n_assets - 1
        found = []
        for name, val in self.kernel_.get_params().items():
            if name.endswith("length_scale"):
                arr = np.atleast_1d(np.asarray(val, dtype=float))
                if arr.size == 1:                 # isotropic: broadcast to n_dims
                    arr = np.full(n_dims, arr[0])
                found.append(arr)
        if not found:
            raise RuntimeError("no length_scale found in fitted kernel")
        return found
 
    def _primary_length_scale(self):
        """
        Returns the single most representative length-scale vector for
        diagnostics: the elementwise geometric mean across all length_scale
        terms present. For a single-ARD kernel this is just that vector; for
        multi-scale kernels it summarizes the combined characteristic scale per
        coordinate. Returns a 1D array of length n_assets-1.
        """
        scales = self._length_scales()
        return np.exp(np.mean(np.log(np.vstack(scales)), axis=0))
 
    def _check_bounds(self):
        """
        Warns when any fitted hyperparameter sits at its optimization bound.
        A pinned value usually means the true optimum lies outside the search
        box (bound too tight), not that the model converged. Emits one warning
        per offending hyperparameter. Returns None.
        """
        params = self.kernel_.get_params()
        for name, val in params.items():
            bkey = name + "_bounds"
            if bkey in params:
                bounds = np.ravel(params[bkey])
                if bounds.size < 2:
                    continue
                lo, hi = float(bounds[0]), float(bounds[1])
                for v in np.ravel(val):
                    if v <= lo * 1.01 or v >= hi * 0.99:
                        warnings.warn(
                            f"hyperparameter {name}={v:.3g} hit bound "
                            f"[{lo:.3g}, {hi:.3g}]; consider widening it.")
                        break
 
    def fit(self, W, y, alpha: Optional[float] = 0.0):
        """
        Fits the surrogate to data points.
        W: portfolio feature data (K x n_assets), K = number of observations.
        y: corresponding measured values (length K) for W.
        alpha: optional per-point or scalar observation variance, added on top of
            the fixed jitter; use it when the measure's estimation error is known.
        Returns self.
        """
        X = self._prepare(W)
        self.gp_ = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha + self.jitter,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state
        )
        self.gp_.fit(X, y)
        self.kernel_ = self.gp_.kernel_
        self.warm = False
        self._check_bounds()
        return self
 
    def warm_start(self):
        """
        Promotes the fitted kernel to the starting point of the next fit.
        Useful when sampling is sequential and adaptive, since readjusting from
        the last best kernel improves processing time and stability.
        Returns self. Raises RuntimeError if called before fit.
        """
        if self.kernel_ is None:
            raise RuntimeError("run fit before warm_start")
        self.kernel = self.kernel_
        self.warm = True
        return self
 
    def predict(self, W, return_std=True):
        """
        Posterior prediction on new portfolios.
        W: portfolio weights (K x n_assets), each row summing to 1.
        return_std: if True, also returns the posterior standard deviation.
        Returns the posterior mean (length K), and the standard deviation when
        return_std is True.
        """
        X = self._prepare(W)
        return self.gp_.predict(X, return_std=return_std)
 
    def sample_y(self, W, n_samples=1):
        """
        Draws sample functions from the posterior.
        W: portfolio weights (K x n_assets), each row summing to 1.
        n_samples: number of posterior functions to draw.
        Returns an array (K x n_samples) of sampled values, useful for
        uncertainty propagation.
        """
        X = self._prepare(W)
        return self.gp_.sample_y(X, n_samples=n_samples,
                                 random_state=self.random_state)
 
    def diagnostics(self):
        """
        Reports the fitted surrogate's health, all cheap to read after fit.
        Returns a dict with:
          length_scales : representative length-scale per projected coordinate
                          (geometric mean across all length_scale terms; smaller
                          = more sensitive; comparable within one model)
          n_scale_terms : how many length_scale terms the kernel has (1 for a
                          single Matern, more for multi-scale kernels)
          amplitude     : signal variance sigma_f^2
          noise         : learned estimation noise sigma_n^2
          snr           : amplitude / noise (low = noise dominates the signal)
          log_marg_like : marginal log-likelihood at the optimum (higher wins
                          when comparing candidate priors on the same data)
        """
        params = self.kernel_.get_params()
        amp = next(v for k, v in params.items() if k.endswith("constant_value"))
        noise = next(v for k, v in params.items() if k.endswith("noise_level"))
        return {
            "length_scales": self._primary_length_scale(),
            "n_scale_terms": len(self._length_scales()),
            "amplitude": float(amp),
            "noise": float(noise),
            "snr": float(amp / noise),
            "log_marg_like": float(self.gp_.log_marginal_likelihood_value_),
        }
 
    def sensitivity_metric(self):
        """
        Maps the length-scales back to the original asset space as a sensitivity
        metric M = U^T Lambda U. Lambda sums diag(1 / l_k^2) over every
        length_scale term in the kernel, so isotropic, single-ARD and
        multi-scale kernels are all handled.
        Returns a dict with:
          M       : n_assets x n_assets symmetric PSD matrix of rank n_assets-1
                    (the zero eigenvalue is the simplex direction sum(w)=1)
          eigvals : eigenvalues in descending order (last one ~ 0)
          eigvecs : matching eigenvectors as columns; the leading one is the
                    portfolio trade the measure reacts to most
          diag    : diagonal of M, a per-asset marginal importance (drops the
                    cross terms M_ij that capture inter-asset interactions)
        """
        scales = self._length_scales()
        Lambda = sum(np.diag(1.0 / s ** 2) for s in scales)
        M = self._U.T @ Lambda @ self._U
        M = 0.5 * (M + M.T)
        eigvals, eigvecs = np.linalg.eigh(M)
        order = np.argsort(eigvals)[::-1]
        return {
            "M": M,
            "eigvals": eigvals[order],
            "eigvecs": eigvecs[:, order],
            "diag": np.diag(M),
        }
 
    def __repr__(self):
        status = "fitted" if self.gp_ is not None else "unfitted"
        return (f"<Object: Model SurrogateGPR> (Objective function={self.sim_obj}; "
                f"status={status}; n_assets={self.n_assets}; warm-start={self.warm})\n"
                f" └── <Kernel: {self.kernel}>")



 
 
class SurrogateKernels:
    """
    Namespace of ready-to-use kernel priors. Each nested class returns a composed
    scikit-learn Kernel when instantiated; see each docstring for the assumptions
    it makes and when to use it.
    """
 
    class Plain:
        """
        Amplitude * Matern 5/2 (ARD) + white noise.
 
        Assumes: a smooth, twice-differentiable stationary signal with no known
        global trend, plus homogeneous estimation noise. Matern 5/2 keeps two
        derivatives (unlike RBF's infinite smoothness, which over-smooths regime
        breaks) while ARD gives each projected coordinate its own length-scale.
        The safe default when the surface's global shape is unknown.
 
        n_dims: number of projected coordinates (n_assets - 1).
        amp_bounds, ls_bounds, noise_bounds: optimization bounds for amplitude,
        length-scales, and noise level.
        Returns a scikit-learn Kernel.
        """
        def __new__(cls, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), noise_bounds=(1e-6, 1e1)):
            return (ConstantKernel(1.0, amp_bounds)
                    * Matern([1.0] * n_dims, ls_bounds, nu=2.5)
                    + WhiteKernel(1e-2, noise_bounds))
 
    class Tail:
        """
        Amplitude * Matern 5/2 (ARD) + linear trend + white noise.
 
        Assumes: a smooth signal riding on a dominant linear trend, plus
        estimation noise. For ratio/tail measures (Sharpe, VaR, CVaR, STARR):
        the Matern captures local curvature and the added DotProduct supplies a
        global direction the Matern alone would not extrapolate, reverting to the
        mean away from data. White noise matters because bootstrap/tail estimates
        are noisy.
 
        n_dims: number of projected coordinates (n_assets - 1).
        amp_bounds, ls_bounds, sigma0_bounds, noise_bounds: optimization bounds.
        Returns a scikit-learn Kernel.
        """
        def __new__(cls, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), sigma0_bounds=(1e-3, 1e3),
                    noise_bounds=(1e-6, 1e1)):
            return (ConstantKernel(1.0, amp_bounds)
                    * Matern([1.0] * n_dims, ls_bounds, nu=2.5)
                    + DotProduct(1.0, sigma0_bounds)
                    + WhiteKernel(1e-2, noise_bounds))
 
    class Markowitz:
        """
        Amplitude * Matern 5/2 (ARD) + quadratic trend + white noise.
 
        Assumes: the measure is dominated by a global quadratic form, exactly the
        case for Markowitz variance w^T S w. The squared DotProduct encodes that
        bowl shape and extrapolates keeping it; the Matern models residual
        departures; white noise absorbs estimation error.
 
        n_dims: number of projected coordinates (n_assets - 1).
        amp_bounds, ls_bounds, sigma0_bounds, noise_bounds: optimization bounds.
        Returns a scikit-learn Kernel.
        """
        def __new__(cls, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), sigma0_bounds=(1e-3, 1e3),
                    noise_bounds=(1e-6, 1e1)):
            return (ConstantKernel(1.0, amp_bounds)
                    * Matern([1.0] * n_dims, ls_bounds, nu=2.5)
                    + DotProduct(1.0, sigma0_bounds) ** 2
                    + WhiteKernel(1e-2, noise_bounds))
 
    class MultiScale:
        """
        Amplitude * RationalQuadratic + white noise.
 
        Assumes: the surface varies at several length-scales at once. The
        RationalQuadratic is an infinite mixture of RBFs over length-scales
        (alpha controls the spread), capturing broad and fine variation together.
        Trade-off: it recovers RBF's infinite smoothness, so for sharp tail
        measures prefer Tail or TwoScaleMatern.
 
        n_dims: accepted for a consistent interface (RQ is isotropic here).
        amp_bounds, ls_bounds, alpha_bounds, noise_bounds: optimization bounds.
        Returns a scikit-learn Kernel.
        """
        def __new__(cls, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2),
                    noise_bounds=(1e-6, 1e1)):
            return (ConstantKernel(1.0, amp_bounds)
                    * RationalQuadratic(1.0, 1.0, ls_bounds, alpha_bounds)
                    + WhiteKernel(1e-2, noise_bounds))
 
    class TwoScaleMatern:
        """
        (Amp * Matern 5/2 long ARD) + (Amp * Matern 5/2 short ARD) + white noise.
 
        Assumes: two additive smooth components, a broad slow variation and a
        fine local detail, each with its own ARD length-scales, while keeping
        finite (twice-differentiable) smoothness. Explicit-control alternative to
        MultiScale for sharp tail measures where RBF-like smoothness is unwanted.
 
        n_dims: number of projected coordinates (n_assets - 1).
        amp_bounds, ls_bounds, noise_bounds: optimization bounds shared by both
        Matern terms and the noise.
        Returns a scikit-learn Kernel.
        """
        def __new__(cls, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), noise_bounds=(1e-6, 1e1)):
            return (ConstantKernel(1.0, amp_bounds)
                    * Matern([3.0] * n_dims, ls_bounds, nu=2.5)
                    + ConstantKernel(0.5, amp_bounds)
                    * Matern([0.3] * n_dims, ls_bounds, nu=2.5)
                    + WhiteKernel(1e-2, noise_bounds))
 