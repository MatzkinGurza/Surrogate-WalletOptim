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


import autograd.numpy as anp

 
 
class SurrogateKernels:
    """
    Namespace of ready-to-use kernel priors. Each nested class returns a composed
    scikit-learn Kernel when instantiated; see each docstring for the assumptions
    it makes and when to use it.
    """
    @staticmethod
    def _matern52(x_h, X_train, amp, length_scales):
        """amp * Matern(nu=5/2, ARD). length_scales: vetor (n_dims,)."""
        diff = (x_h[None, :] - X_train) / length_scales[None, :]
        r2 = anp.sum(diff * diff, axis=1)
        r = anp.sqrt(r2 + 1e-12)
        s5 = anp.sqrt(5.0)
        return amp * (1.0 + s5 * r + (5.0 / 3.0) * r2) * anp.exp(-s5 * r)
    
    @staticmethod
    def _dot(x_h, X_train, sigma_0):
        """DotProduct: sigma_0^2 + x.x'."""
        return sigma_0 ** 2 + X_train @ x_h
    
    @staticmethod
    def _rational_quadratic(x_h, X_train, amp, length_scale, alpha_rq):
        """amp * RationalQuadratic isotropico: amp*(1 + d^2/(2 alpha l^2))^(-alpha)."""
        diff = x_h[None, :] - X_train
        d2 = anp.sum(diff * diff, axis=1)
        return amp * (1.0 + d2 / (2.0 * alpha_rq * length_scale ** 2)) ** (-alpha_rq)
    
    
    
 
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
        def __init__(self, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), noise_bounds=(1e-10, 1e1)):
            self.n_dims = n_dims
            self.amp_bounds = amp_bounds
            self.ls_bounds = ls_bounds
            self.noise_bounds = noise_bounds

        def build(self):
            return (ConstantKernel(1.0, self.amp_bounds)
                    * Matern([1.0] * self.n_dims, self.ls_bounds, nu=2.5)
                    + WhiteKernel(1e-2, self.noise_bounds))

        @staticmethod
        def eval_autograd(x, X_train, hp):
            """
            Evaluates the Plain kernel using autograd for gradient computation.
            x: 1D array of shape (n_dims,) representing a single input point.
            X_train: 2D array of shape (n_samples, n_dims) representing the training data.
            hp: dictionary containing hyperparameters:
                - 'amp': amplitude (float)
                - 'length_scales': length scale (float)
            """
            anp_kernel = SurrogateKernels._matern52(x, X_train, hp['amp'], hp['length_scales'])
            return anp_kernel


        def _read_params(self, kernel_origin, p:Dict[str, Any], ndims: int) -> Dict[str, Any]:
            """
            Reads the hyperparameters from a dictionary and returns a new dictionary
            with the relevant parameters for the Plain kernel (used on .eval_autograd method as parameter hp).
            p: dictionary containing hyperparameters returned from kernel.get_params().
            Returns a dictionary with keys 'amp' and 'length_scales'.
            """
            if kernel_origin.__class__ != self.__class__:
                raise ValueError(f"kernel_origin must be an instance of {self.__class__.__name__}")
            else:
                try:
                    amp = p['constant_value']
                    length_scales = p['length_scale']
                    if isinstance(length_scales, float):
                        length_scales = np.full(ndims, length_scales)
                    elif isinstance(length_scales, list):
                        length_scales = np.array(length_scales)
                    elif isinstance(length_scales, np.ndarray):
                        pass
                    else:
                        raise ValueError(f"Unexpected type for length_scales: {type(length_scales)}")

                    return {'amp': amp, 'length_scales': length_scales}
                    
                except KeyError as e:
                    raise KeyError(f"Missing expected hyperparameter key: {e}")
                except Exception as e:
                    raise ValueError(f"Error reading hyperparameters: {e}")



 
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
        def __init__(self, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), sigma0_bounds=(1e-6, 1e3),
                    noise_bounds=(1e-10, 1e1)):
            self.n_dims = n_dims
            self.amp_bounds = amp_bounds
            self.ls_bounds = ls_bounds
            self.sigma0_bounds = sigma0_bounds
            self.noise_bounds = noise_bounds

        def build(self):
            return (ConstantKernel(1.0, self.amp_bounds)
                    * Matern([1.0] * self.n_dims, self.ls_bounds, nu=2.5)
                    + DotProduct(1.0, self.sigma0_bounds)
                    + WhiteKernel(1e-2, self.noise_bounds))

        @staticmethod
        def eval_autograd(x, X_train, hp):
            """
            Evaluates the Tail kernel using autograd for gradient computation.
            x: 1D array of shape (n_dims,) representing a single input point.
            X_train: 2D array of shape (n_samples, n_dims) representing the training data.
            hp: dictionary containing hyperparameters:
                - 'amp': amplitude (float)
                - 'length_scales': length scale (float)
                - 'sigma_0': linear trend coefficient (float)
            """
            anp_kernel = (SurrogateKernels._matern52(x, X_train, hp["amp"], hp["length_scales"])
            + SurrogateKernels._dot(x, X_train, hp["sigma_0"]))
            return anp_kernel

        def _read_params(self, kernel_origin, p:Dict[str, Any], ndims: int) -> Dict[str, Any]:
            """
            Reads the hyperparameters from a dictionary and returns a new dictionary
            with the relevant parameters for the Tail kernel (used on .eval_autograd method as parameter hp).
            p: dictionary containing hyperparameters returned from kernel.get_params().
            Returns a dictionary with keys 'amp', 'length_scales', and 'sigma_0'.
            """
            if kernel_origin.__class__ != self.__class__:
                raise ValueError(f"kernel_origin must be an instance of {self.__class__.__name__}")
            else:
                try:
                    amp = p['constant_value']
                    length_scales = p['length_scale']
                    sigma_0 = p['sigma_0']
                    if isinstance(length_scales, float):
                        length_scales = np.full(ndims, length_scales)
                    elif isinstance(length_scales, list):
                        length_scales = np.array(length_scales)
                    elif isinstance(length_scales, np.ndarray):
                        pass
                    else:
                        raise ValueError(f"Unexpected type for length_scales: {type(length_scales)}")

                    return {'amp': amp, 'length_scales': length_scales, 'sigma_0': sigma_0}
                    
                except KeyError as e:
                    raise KeyError(f"Missing expected hyperparameter key: {e}")
                except Exception as e:
                    raise ValueError(f"Error reading hyperparameters: {e}")
 
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
        def __init__(self, n_dims, amp_bounds=(1e-3, 1e3),
                    ls_bounds=(1e-2, 1e2), sigma0_bounds=(1e-6, 1e3),
                    noise_bounds=(1e-10, 1e1)):
            self.n_dims = n_dims
            self.amp_bounds = amp_bounds
            self.ls_bounds = ls_bounds
            self.sigma0_bounds = sigma0_bounds
            self.noise_bounds = noise_bounds

        def build(self):
            return (ConstantKernel(1.0, self.amp_bounds)
                    * Matern([1.0] * self.n_dims, self.ls_bounds, nu=2.5)
                    + DotProduct(1.0, self.sigma0_bounds) ** 2
                    + WhiteKernel(1e-2, self.noise_bounds))

        @staticmethod
        def eval_autograd(x, X_train, hp):
            """
            Evaluates the Markowitz kernel using autograd for gradient computation.
            x: 1D array of shape (n_dims,) representing a single input point.
            X_train: 2D array of shape (n_samples, n_dims) representing the training data.
            hp: dictionary containing hyperparameters:
                - 'amp': amplitude (float)
                - 'length_scales': length scale (float)
                - 'sigma_0': linear trend coefficient (float)
            """
            anp_kernel = (SurrogateKernels._matern52(x, X_train, hp["amp"], hp["length_scales"])
            + SurrogateKernels._dot(x, X_train, hp["sigma_0"])**2)
            return anp_kernel

        def _read_params(self, kernel_origin, p:Dict[str, Any], ndims: int) -> Dict[str, Any]:
            """
            Reads the hyperparameters from a dictionary and returns a new dictionary
            with the relevant parameters for the Markowitz kernel (used on .eval_autograd method as parameter hp).
            p: dictionary containing hyperparameters returned from kernel.get_params().
            Returns a dictionary with keys 'amp', 'length_scales', and 'sigma_0'.
            """
            if kernel_origin.__class__ != self.__class__:
                raise ValueError(f"kernel_origin must be an instance of {self.__class__.__name__}")
            else:
                try:
                    amp = p['constant_value']
                    length_scales = p['length_scale']
                    sigma_0 = p['sigma_0']
                    if isinstance(length_scales, float):
                        length_scales = np.full(ndims, length_scales)
                    elif isinstance(length_scales, list):
                        length_scales = np.array(length_scales)
                    elif isinstance(length_scales, np.ndarray):
                        pass
                    else:
                        raise ValueError(f"Unexpected type for length_scales: {type(length_scales)}")

                    return {'amp': amp, 'length_scales': length_scales, 'sigma_0': sigma_0}
                    
                except KeyError as e:
                    raise KeyError(f"Missing expected hyperparameter key: {e}")
                except Exception as e:
                    raise ValueError(f"Error reading hyperparameters: {e}")

 
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
        def __init__(self, n_dims, amp_bounds=(1e-3, 1e3),
                     ls_bounds=(1e-2, 1e2), alpha_bounds=(1e-2, 1e2),
                     noise_bounds=(1e-10, 1e1)):
            self.n_dims = n_dims
            self.amp_bounds = amp_bounds
            self.ls_bounds = ls_bounds
            self.alpha_bounds = alpha_bounds
            self.noise_bounds = noise_bounds

        def build(self):
            return (ConstantKernel(1.0, self.amp_bounds)
                    * RationalQuadratic(1.0, 1.0, self.ls_bounds, self.alpha_bounds)
                    + WhiteKernel(1e-2, self.noise_bounds))

        @staticmethod
        def eval_autograd(x, X_train, hp):
            """
            Evaluates the MultiScale kernel using autograd for gradient computation.
            x: 1D array of shape (n_dims,) representing a single input point.
            X_train: 2D array of shape (n_samples, n_dims) representing the training data.
            hp: dictionary containing hyperparameters:
                - 'amp': amplitude (float)
                - 'length_scale_iso': length scale (float)
                - 'alpha_rq': alpha parameter for RationalQuadratic (float)
            """
            anp_kernel = SurrogateKernels._rational_quadratic(x, X_train, hp["amp"],
                               hp["length_scale_iso"], hp["alpha_rq"])
            return anp_kernel

        def _read_params(self, kernel_origin, p:Dict[str, Any], ndims: int) -> Dict[str, Any]:
            """
            Reads the hyperparameters from a dictionary and returns a new dictionary
            with the relevant parameters for the MultiScale kernel (used on .eval_autograd method as parameter hp).
            p: dictionary containing hyperparameters returned from kernel.get_params().
            Returns a dictionary with keys 'amp', 'length_scale_iso', and 'alpha_rq'.
            """
            if kernel_origin.__class__ != self.__class__:
                raise ValueError(f"kernel_origin must be an instance of {self.__class__.__name__}")
            else:
                try:
                    amp = p['constant_value']
                    length_scale_iso = p['length_scale']
                    alpha_rq = p['alpha']
                    return {'amp': amp, 'length_scale_iso': length_scale_iso, 'alpha_rq': alpha_rq}
                except KeyError as e:
                    raise KeyError(f"Missing expected hyperparameter key: {e}")
                except Exception as e:
                    raise ValueError(f"Error reading hyperparameters: {e}")
 
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
        def __init__(self, n_dims, amp_bounds=(1e-3, 1e3),
                     ls_bounds=(1e-2, 1e2), noise_bounds=(1e-10, 1e1)):
            self.n_dims = n_dims
            self.amp_bounds = amp_bounds
            self.ls_bounds = ls_bounds
            self.noise_bounds = noise_bounds

        def build(self):
            return (ConstantKernel(1.0, self.amp_bounds)
                    * Matern([3.0] * self.n_dims, self.ls_bounds, nu=2.5)
                    + ConstantKernel(0.5, self.amp_bounds)
                    * Matern([0.3] * self.n_dims, self.ls_bounds, nu=2.5)
                    + WhiteKernel(1e-2, self.noise_bounds))  

        @staticmethod
        def eval_autograd(x, X_train, hp):
            """
            Evaluates the TwoScaleMatern kernel using autograd for gradient computation.
            x: 1D array of shape (n_dims,) representing a single input point.
            X_train: 2D array of shape (n_samples, n_dims) representing the training data.
            hp: dictionary containing hyperparameters:
                - 'amp_long': amplitude for long Matern (float)
                - 'length_scales_long': length scale for long Matern (float)
                - 'amp_short': amplitude for short Matern (float)
                - 'length_scales_short': length scale for short Matern (float)
            """
            anp_kernel = (SurrogateKernels._matern52(x, X_train, hp["amp_long"], hp["length_scales_long"])
                          + SurrogateKernels._matern52(x, X_train, hp["amp_short"], hp["length_scales_short"]))
            return anp_kernel

        def _read_params(self, kernel_origin, p:Dict[str, Any], ndims: int) -> Dict[str, Any]:
            """
            Reads the hyperparameters from a dictionary and returns a new dictionary
            with the relevant parameters for the TwoScaleMatern kernel (used on .eval_autograd method as parameter hp).
            p: dictionary containing hyperparameters returned from kernel.get_params().
            Returns a dictionary with keys 'amp_long', 'length_scales_long', 'amp_short', and 'length_scales_short'.
            """
            if kernel_origin.__class__ != self.__class__:
                raise ValueError(f"kernel_origin must be an instance of {self.__class__.__name__}")
            else:
                try:
                    amp_long = p['constant_value']
                    length_scales_long = p['length_scale']
                    amp_short = p['constant_value_1']
                    length_scales_short = p['length_scale_1']
                    if isinstance(length_scales_long, float):
                        length_scales_long = np.full(ndims, length_scales_long)
                    elif isinstance(length_scales_long, list):
                        length_scales_long = np.array(length_scales_long)
                    elif isinstance(length_scales_long, np.ndarray):
                        pass
                    else:
                        raise ValueError(f"Unexpected type for length_scales_long: {type(length_scales_long)}")

                    if isinstance(length_scales_short, float):
                        length_scales_short = np.full(ndims, length_scales_short)
                    elif isinstance(length_scales_short, list):
                        length_scales_short = np.array(length_scales_short)
                    elif isinstance(length_scales_short, np.ndarray):
                        pass
                    else:
                        raise ValueError(f"Unexpected type for length_scales_short: {type(length_scales_short)}")

                    return {
                        'amp_long': amp_long,
                        'length_scales_long': length_scales_long,
                        'amp_short': amp_short,
                        'length_scales_short': length_scales_short
                    }
                    
                except KeyError as e:
                    raise KeyError(f"Missing expected hyperparameter key: {e}")
                except Exception as e:
                    raise ValueError(f"Error reading hyperparameters: {e}")





kernel_classes = Union[SurrogateKernels.Markowitz, 
                       SurrogateKernels.Plain, 
                       SurrogateKernels.Tail, 
                       SurrogateKernels.TwoScaleMatern, 
                       SurrogateKernels.MultiScale]




   

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
 
    def __init__(self, kernel_obj: kernel_classes,
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
        self.kernel_obj = kernel_obj            # initial prior
        self.kernel = kernel_obj.build()        # initial prior, ready for sklearn
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


