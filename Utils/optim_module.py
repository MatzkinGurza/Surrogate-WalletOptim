import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Literal, Optional, Tuple, Union, Any, Type
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


from surrogate_module import SurrogateModel, SurrogateKernels
import autograd.numpy as anp
from autograd import grad as _agrad


kernel_type = Union[SurrogateKernels.Markowitz, 
                    SurrogateKernels.Plain, 
                    SurrogateKernels.Tail, 
                    SurrogateKernels.TwoScaleMatern, 
                    SurrogateKernels.MultiScale]


class EFOptimizer:
    """
    Efficient Frontier Optimizer class for Gaussian Process Regressor and a flexible kernel class.
    Method: 
    Optimizes through gradient descent working with the fitted parameters of the GPR and its kernel.
    Descent is restricted to simplex by running on the corresponding Helmert space intrinsic to the SurrogatModel, is limited to positiv values and is regularized for portfolio returns restriction.
    Regularization is done through Lagrange penalty method to avoid complex manual tuning of lambda penalty parameter.
    Uses autograd to compute gradients directly related to the kernel hyperparameters and the GPR fitted parameters.
    For a given G level of return, the optimizer minimizes the riks function approximated by the GPR model and its kernel, while ensuring that the portfolio weights sum to 1 and are non-negative.
    Optimizer can adapt for G restirction as inequality or equality constraint, as well as no G constraint on returns.
    """
    def __init__(self, surrogate: SurrogateModel, kernel_cls: kernel_type):
        """
        Initializes the EFOptimizer with a Gaussian Process model and a kernel class.
        """
        # building blocks
        self.surrogate = surrogate
        self.ndims = surrogate.n_assets - 1 
        self.gp_ = surrogate.gp_
        self.kernel_cls = kernel_cls
        self.kernel_ = self.gp_.kernel_
        self.params_ = self.gp_.kernel_.get_params()
        self.hp_ = kernel_cls._read_params(kernel_origin= surrogate.kernel_obj, p=self.params_, ndims=self.ndims)
        # constants
        self._Xtr    = self.gp_.X_train_          # (N, n_dims) já no espaço de Helmert
        self._alpha  = self.gp_.alpha_.ravel()    # (N,) coeficientes duais
        self._y_std  = self.gp_._y_train_std      # normalize_y: desvio
        self._y_mean = self.gp_._y_train_mean     # normalize_y: média
        # automatic gradient from surface with autograd on x_h (helmert space)
        self._surface_grad = _agrad(self.surface)

        def surface(self, x_h):
            """
            GPR posterior average on x_h point from helmert space.
            Closed formula is differentiable: y_std * (alpha . k(x_h, X_train)) + y_mean.
            The eval_autograd of the kernel reconstructs only the signal terms (without white noise),
            which coincides with the mean outside the training points.
            """
            k_vec = self.kernel_cls.eval_autograd(x_h, self._Xtr, self.hp_)
            return self._y_std * anp.dot(self._alpha, k_vec) + self._y_mean

        def surface_grad(self, x_h):
            """
            Surface gradient in relation to x_h (Helmert space), via autograd.
            """
            return self._surface_grad(anp.asarray(x_h, dtype=float))
        

        
