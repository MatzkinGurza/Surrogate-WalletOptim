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


class SurrogateGPR:
    def __init__(self, kernel: Kernel, 
                 n_assets: int, 
                 n_restarts: int, 
                 sim_obj: Optional[str], 
                 random_state: int = 0 ,
                 alpha_jitter: float = 5e-7):
        self.kernel = kernel                # prior inicial
        self.kernel_ = None                 # kernel ajustado, preenchido depois do fit
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.n_assets = n_assets
        self._U = helmert(n_assets)         # base de Helmert utilizada para transformação isométrica evitando degeneração
        self.gp_ = None                     # guarda o regressor após o fit
        self.jitter = alpha_jitter          # jitter evita matriz não invertível se pontos estão muito próximos uns dos outros


    def _prepare(self, W):
        try:
            W = np.asarray(W, dtype=float)
            return W @ self._U.T
        except Exception as e:
            raise ValueError(f"_prepare exited with exception: {e} for values W={W}, self._U={self._U}")

    def fit(self, W, y, alpha: Optional[float|np.array]=0.0):
        """
        Ajusts surrogate to data points. 
        W: portfolio feature data (K x n_assets) where K is the number of observations. 
        y: Corresponding measured values (length K) for W.
        """
        self.gp_ = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha+self.jitter,
            normalize_y=True,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.random_state
        )
        self.kernel_ = self.gp_.kernel_
        self.check_bounds() !!!!!!
        return self

    def warm_start(self):
        """
        Promotes adjusted kernel to warm starting point for next fit call. 
        Useful in cases where sampling is sequential and adaptative due to readjustments 
        being made over last best kernel. Helps with processing time and stability.
        """
        if self.kernel_ is None:
            raise RuntimeError("rode fit antes de warm_start")
        self.kernel = self.kernel_
        return self

    def predict(self, W, return_std):
        """Média posterior (e desvio-padrão) em novas carteiras."""
        X = self._prepare(W)
        return self.gp_.predict(X, return_std=return_std)

    