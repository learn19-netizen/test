"""
Data Loader Module
Downloads historical price data from Yahoo Finance and performs data cleaning.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


class DataLoader:
    """Handles data downloading and cleaning from Yahoo Finance."""

    def __init__(self, tickers: List[str], years_of_history: int = 10):
        """
        Initialize the DataLoader.

        Args:
            tickers: List of ticker symbols to download
            years_of_history: Number of years of historical data to fetch
        """
        self.tickers = tickers
        self.years_of_history = years_of_history
        self.raw_data: Optional[pd.DataFrame] = None
        self.clean_data: Optional[pd.DataFrame] = None
        self.returns: Optional[pd.DataFrame] = None

    def download_data(self, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Download historical adjusted close prices from Yahoo Finance.

        Args:
            end_date: End date for data download (defaults to today)

        Returns:
            DataFrame with adjusted close prices for all tickers
        """
        if end_date is None:
            end_date = datetime.now()

        start_date = end_date - timedelta(days=self.years_of_history * 365)

        print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Tickers: {self.tickers}")

        # Download data using yfinance
        data = yf.download(
            self.tickers,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            auto_adjust=True,
            progress=True
        )

        # Extract Close prices (already adjusted due to auto_adjust=True)
        if isinstance(data.columns, pd.MultiIndex):
            self.raw_data = data['Close']
        else:
            # Single ticker case
            self.raw_data = data[['Close']]
            self.raw_data.columns = self.tickers

        print(f"Downloaded {len(self.raw_data)} rows of data")
        return self.raw_data

    def remove_missing_values(self, data: pd.DataFrame,
                               max_missing_pct: float = 0.1) -> pd.DataFrame:
        """
        Remove columns with too many missing values and forward-fill remaining NAs.

        Args:
            data: Input DataFrame
            max_missing_pct: Maximum allowed percentage of missing values per column

        Returns:
            Cleaned DataFrame
        """
        # Calculate missing percentage per column
        missing_pct = data.isnull().sum() / len(data)

        # Report tickers with high missing values
        high_missing = missing_pct[missing_pct > max_missing_pct]
        if len(high_missing) > 0:
            print(f"Removing tickers with >{max_missing_pct*100}% missing values:")
            for ticker, pct in high_missing.items():
                print(f"  {ticker}: {pct*100:.1f}% missing")

        # Keep only columns with acceptable missing values
        valid_columns = missing_pct[missing_pct <= max_missing_pct].index
        data = data[valid_columns]

        # Forward fill then backward fill remaining NAs
        data = data.ffill().bfill()

        # Drop any remaining rows with NAs
        data = data.dropna()

        return data

    def remove_outliers(self, data: pd.DataFrame,
                        n_std: float = 4.0) -> pd.DataFrame:
        """
        Remove outliers based on z-score of returns.

        Args:
            data: Input price DataFrame
            n_std: Number of standard deviations for outlier threshold

        Returns:
            DataFrame with outliers replaced by interpolated values
        """
        # Calculate returns
        returns = data.pct_change()

        # For each column, identify outliers
        for col in returns.columns:
            col_returns = returns[col].dropna()
            mean = col_returns.mean()
            std = col_returns.std()

            # Identify outlier indices
            outliers = np.abs(returns[col] - mean) > n_std * std
            n_outliers = outliers.sum()

            if n_outliers > 0:
                print(f"  {col}: {n_outliers} outliers detected (>{n_std} std)")
                # Replace outliers with NaN and interpolate
                data.loc[outliers, col] = np.nan

        # Interpolate to fill replaced outliers
        data = data.interpolate(method='linear').ffill().bfill()

        return data

    def clean_data_pipeline(self, max_missing_pct: float = 0.1,
                            outlier_std: float = 4.0) -> pd.DataFrame:
        """
        Run the full data cleaning pipeline.

        Args:
            max_missing_pct: Maximum allowed percentage of missing values
            outlier_std: Number of standard deviations for outlier detection

        Returns:
            Cleaned price DataFrame
        """
        if self.raw_data is None:
            raise ValueError("No data downloaded. Call download_data() first.")

        print("\n--- Cleaning Data ---")

        # Step 1: Remove missing values
        print("Step 1: Handling missing values...")
        data = self.remove_missing_values(self.raw_data.copy(), max_missing_pct)

        # Step 2: Remove outliers
        print("Step 2: Removing outliers...")
        data = self.remove_outliers(data, outlier_std)

        self.clean_data = data
        print(f"Clean data shape: {data.shape}")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")

        return self.clean_data

    def calculate_returns(self, log_returns: bool = True) -> pd.DataFrame:
        """
        Calculate returns from clean price data.

        Args:
            log_returns: If True, calculate log returns; else simple returns

        Returns:
            DataFrame of returns
        """
        if self.clean_data is None:
            raise ValueError("No clean data. Run clean_data_pipeline() first.")

        if log_returns:
            self.returns = np.log(self.clean_data / self.clean_data.shift(1))
        else:
            self.returns = self.clean_data.pct_change()

        self.returns = self.returns.dropna()
        return self.returns

    def get_statistics(self) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Calculate mean returns and covariance matrix.

        Returns:
            Tuple of (mean_returns, covariance_matrix)
        """
        if self.returns is None:
            self.calculate_returns()

        # Annualize (assuming daily data, ~252 trading days)
        mean_returns = self.returns.mean() * 252
        cov_matrix = self.returns.cov() * 252

        return mean_returns, cov_matrix

    def get_available_tickers(self) -> List[str]:
        """Return list of tickers that survived the cleaning process."""
        if self.clean_data is None:
            return []
        return list(self.clean_data.columns)


def create_default_loader() -> DataLoader:
    """Create a DataLoader with the default universe of assets."""
    tickers = [
        # Thai Export
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        # Thai Domestic
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        # Global
        'WDC', 'THD',
        # Fixed Income
        'LEMB', 'VWOB', 'EMLC',
        # FX
        'THB=X'
    ]
    return DataLoader(tickers, years_of_history=10)


if __name__ == "__main__":
    # Test the data loader
    loader = create_default_loader()
    loader.download_data()
    loader.clean_data_pipeline()
    returns = loader.calculate_returns()
    mean_ret, cov = loader.get_statistics()

    print("\n--- Statistics ---")
    print("Annualized Mean Returns:")
    print(mean_ret)
    print("\nCovariance Matrix Shape:", cov.shape)
