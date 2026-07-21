import pandas as pd
import numpy as np
from typing import List, Dict, Any, Literal, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StockPortfolio:
    """A class to represent a stock portfolio </p>Attributes:</p>
    stocks (Tuple[str]): A tuple of stock ticker symbols in the portfolio.</p>
    composition (Tuple[float]): A tuple of corresponding weights for each stock in the portfolio.
    """
    stocks: Tuple[str] = field(default_factory=tuple) #tickers should not be mutable, therefore justifies using tuple instead of list
    composition: Tuple[float] = field(default_factory=tuple) #weights should not be mutable, therefore justifies using tuple instead of list
    assert sum(composition) == 1.0, "Composition weights must sum to 1.0."

    def alter_composition(self, new_composition: Tuple[float]) -> None:
        """
        Alter the composition of the portfolio.</p>Parameters:</p>
        new_composition (Tuple[float]): A tuple of new weights for each stock in the portfolio.
        """
        if len(new_composition) != len(self.stocks):
            raise ValueError("New composition must have the same length as the number of stocks.")
        self.composition = new_composition





class MarkowitzCov:
    def __init__(self, n_assets: int):
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        self.n_assets = n_assets

    def __repr__(self):
        return f"<Object: Callable MarkowitzCov> (n_assets={self.n_assets})"


    def Covariance(self, A):
        ...
    
    def __call__(self, X):
        ...




class SharpeIndex:
    def __init__(self, n_assets: int, rf: float):
        assert rf > 0, f"rf must be > 0 (current is {rf})"
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        self.n_assets = n_assets
        self.rf = rf
    
    def __repr__(self):
        return f"<Object: Callable SharpeIndex> (n_assets={self.n_assets}; rf={self.rf})"




class VaR:
    def __init__(self, n_assets: int, 
                 alpha:float, 
                 type:Literal["non-parametric", "parametric", "linearized"]="non-parametric"):
        assert 0<=alpha<=1, f"alpha must be between 0 and 1 (current is {alpha})"
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        assert type in ["non-parametric", "parametric", "linearized"], "type must be one of Literal['non-parametric', 'parametric', 'linearized']"
        self.n_assets = n_assets
        self.alpha = alpha
        self.type = type

    def __repr__(self):
        return f"<Object: Callable VaR> (n_assets={self.n_assets}; alpha={self.alpha}; type={type})"
    
    def order_distibution(R_dist):
        ...
    
    def __call__(self, X):
        ...




class CVaR(VaR):
    def __init__(self, n_assets: int, 
                 alpha:float, 
                 type: Literal["non-parametric", "parametric", "linearized"]="non-parametric"):
        assert 0<=alpha<=1, f"alpha must be between 0 and 1 (current is {alpha})"
        assert n_assets > 0, f"n_assets must be > 0 (current is {n_assets})"
        assert type in ["non-parametric", "parametric", "linearized"], "type must be one of Literal['non-parametric', 'parametric', 'linearized']"
        super().__init__(n_assets, alpha, type)
        self.n_assets = n_assets
        self.alpha = alpha
        self.type = type
    
    def getVaR(self, X):
        return super().__call__(X)

    def __repr__(self):
        return f"<Object: Callable CVaR> (n_assets={self.n_assets}; alpha={self.alpha}; type={type})"
    
    def __call__(self, X):
        ...

