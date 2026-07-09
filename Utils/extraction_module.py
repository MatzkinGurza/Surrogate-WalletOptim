import pickle
from pathlib import Path
from typing import List, Dict, Any, Literal, Optional

import yfinance as yf
import pandas as pd


class StockDataFetcher:
    """A class to fetch historical stock data using the yfinance library."""

    def __init__(self, start_date: str, end_date: str, interval: Literal['1d', '1wk', '1mo'] = '1d'):
        self.start_date = start_date
        self.end_date = end_date
        self.interval = interval
        self.fetcher_cache = {}

    def fetch_data(self, ticker: str, auto_adjust=True) -> pd.DataFrame:
        """
        Fetch historical stock data for a given ticker symbol.

        Parameters:
        ticker (str): The ticker symbol of the stock.
        auto_adjust (bool): Whether to adjust the stock data for splits and dividends.

        Returns:
        pd.DataFrame: A DataFrame containing the historical stock data.
        """
        if (ticker, auto_adjust) in self.fetcher_cache:
            return self.fetcher_cache[(ticker, auto_adjust)].copy()
        try:
            data = yf.download(ticker,
                               start=self.start_date,
                               end=self.end_date,
                               interval=self.interval,
                               auto_adjust=auto_adjust,
                               progress=False,
                               group_by='column')
            self.fetcher_cache[(ticker, auto_adjust)] = data
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            data = pd.DataFrame()
        if data.empty or data is None:
                print(f"No data found for {ticker}.")
        return data.copy()

    def fetch_batch(self, tickers: List[str], auto_adjust=True) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple tickers in a single batched request,
        reusing cached data and only downloading what's missing.

        Parameters:
        tickers (List[str]): The ticker symbols to fetch.
        auto_adjust (bool): Whether to adjust the stock data for splits and dividends.

        Returns:
        Dict[str, pd.DataFrame]: Mapping from ticker to its historical data.
        """
        to_fetch = [t for t in tickers if (t, auto_adjust) not in self.fetcher_cache]
        if to_fetch:
            try:
                batch = yf.download(to_fetch,
                                    start=self.start_date,
                                    end=self.end_date,
                                    interval=self.interval,
                                    auto_adjust=auto_adjust,
                                    progress=False,
                                    group_by='ticker')
            except Exception as e:
                print(f"Error fetching batch data for {to_fetch}: {e}")
                batch = pd.DataFrame()

            for ticker in to_fetch:
                if isinstance(batch.columns, pd.MultiIndex):
                    ticker_data = batch[ticker].dropna(how='all') if ticker in batch.columns.get_level_values(0) else pd.DataFrame()
                elif len(to_fetch) == 1 and not batch.empty:
                    ticker_data = batch.copy()
                else:
                    ticker_data = pd.DataFrame()
                if ticker_data.empty:
                    print(f"No data found for {ticker}.")
                self.fetcher_cache[(ticker, auto_adjust)] = ticker_data

        return {t: self.fetcher_cache[(t, auto_adjust)].copy() for t in tickers}

    def clear_cache(self):
        """Clear the fetcher cache."""
        self.fetcher_cache.clear()


class StockDataWarehouse:
    """A class to manage a collection of stock data fetched using StockDataFetcher."""

    PRICE_COLUMN = "Close"  # fixed because auto_adjust=True

    def __init__(self, fetcher: StockDataFetcher):
        self.fetcher = fetcher
        self.data_store: Dict[str, pd.DataFrame] = {}
        self._order: List[str] = []  # To maintain the order of added stocks

    def add_stock(self, ticker: str, auto_adjust=True) -> bool:
        """
        Add a stock's historical data to the warehouse.

        Parameters:
        ticker (str): The ticker symbol of the stock.
        auto_adjust (bool): Whether to adjust the stock data for splits and dividends.
        """
        if ticker in self.data_store:
            print(f"Data for {ticker} already exists in the warehouse.")
            return True
        data = self.fetcher.fetch_data(ticker, auto_adjust=auto_adjust)
        if not data.empty:
            self.data_store[ticker] = data
            self._order.append(ticker)
            return True
        else:
            print(f"No data found for {ticker}.")
            return False

    def add_stocks(self, tickers: List[str], auto_adjust=True) -> List[str]:
        """Add multiple stocks to the warehouse using a single batched fetch, returning the list of failed additions."""
        to_fetch = []
        for ticker in tickers:
            if ticker in self.data_store:
                print(f"Data for {ticker} already exists in the warehouse.")
            else:
                to_fetch.append(ticker)

        batch = self.fetcher.fetch_batch(to_fetch, auto_adjust=auto_adjust)
        failed = []
        for ticker in to_fetch:
            data = batch.get(ticker)
            if data is not None and not data.empty:
                self.data_store[ticker] = data
                self._order.append(ticker)
            else:
                print(f"No data found for {ticker}.")
                failed.append(ticker)
        return failed

    def get_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Retrieve historical stock data for a given ticker symbol from the warehouse."""
        data = self.data_store.get(ticker)
        return None if data is None else data.copy()

    def list_stocks(self) -> List[str]:
        """Lists stocks in the order of insertion."""
        return list(self._order)

    def price_panel(self) -> pd.DataFrame:
        """Return a DataFrame with the adjusted prices of all stocks, aligned by date."""
        if not self._order:
            raise ValueError("Nenhum ativo carregado.")

        series = {}
        for ticker in self._order:
            col = self.data_store[ticker][self.PRICE_COLUMN]
            if isinstance(col, pd.DataFrame):  # caso de MultiIndex residual
                col = col.iloc[:, 0]
            series[ticker] = col

        panel = pd.DataFrame(series)[self._order]
        return panel.dropna(how="any")

    def save(self, path: str) -> None:
        """Persist the warehouse's stock data and insertion order to disk."""
        payload = {"data_store": self.data_store, "order": self._order}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str) -> None:
        """Load previously saved stock data and insertion order from disk."""
        with open(path, "rb") as f:
            payload = pickle.load(f)
        self.data_store = payload["data_store"]
        self._order = payload["order"]
