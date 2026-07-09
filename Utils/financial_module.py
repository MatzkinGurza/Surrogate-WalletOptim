import pandas as pd
import numpy as np
from typing import List, Dict, Any, Literal, Optional, Tuple
from dataclass import dataclass, field


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
        Alter the composition of the portfolio.

        Parameters:
        new_composition (Tuple[float]): A tuple of new weights for each stock in the portfolio.
        """
        if len(new_composition) != len(self.stocks):
            raise ValueError("New composition must have the same length as the number of stocks.")
        self.composition = new_composition