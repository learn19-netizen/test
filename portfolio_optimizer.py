"""
Portfolio Optimizer Module
Implements mean-variance optimization using scipy.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple


class PortfolioOptimizer:
    """Mean-variance portfolio optimization with constraints."""

    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                 risk_free_rate: float = 0.02):
        """
        Initialize the optimizer.

        Args:
            mean_returns: Expected annualized returns for each asset
            cov_matrix: Annualized covariance matrix
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(mean_returns)
        self.tickers = list(mean_returns.index)

        # Validate inputs
        if len(cov_matrix) != self.n_assets:
            raise ValueError("Covariance matrix dimensions must match number of assets")

    def portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate expected portfolio return."""
        return np.sum(self.mean_returns.values * weights)

    def portfolio_volatility(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility (standard deviation)."""
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))

    def portfolio_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio."""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.risk_free_rate) / vol if vol > 0 else 0

    def negative_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Negative Sharpe ratio for minimization."""
        return -self.portfolio_sharpe_ratio(weights)

    def optimize_max_sharpe(self, allow_short: bool = False) -> Dict:
        """
        Find the portfolio with maximum Sharpe ratio.

        Args:
            allow_short: Whether to allow short selling

        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        # Initial guess: equal weights
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        # Constraints: weights sum to 1
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        # Bounds: no short selling if allow_short=False
        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Optimize
        result = minimize(
            self.negative_sharpe_ratio,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            print(f"Warning: Optimization may not have converged: {result.message}")

        optimal_weights = result.x

        # Round very small weights to zero
        optimal_weights[np.abs(optimal_weights) < 1e-6] = 0
        optimal_weights = optimal_weights / np.sum(optimal_weights)  # Renormalize

        return self._create_result_dict(optimal_weights)

    def optimize_min_variance(self, allow_short: bool = False) -> Dict:
        """
        Find the minimum variance portfolio.

        Args:
            allow_short: Whether to allow short selling

        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        # Minimize variance (volatility squared)
        def variance(weights):
            return self.portfolio_volatility(weights) ** 2

        result = minimize(
            variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            print(f"Warning: Optimization may not have converged: {result.message}")

        optimal_weights = result.x
        optimal_weights[np.abs(optimal_weights) < 1e-6] = 0
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        return self._create_result_dict(optimal_weights)

    def optimize_target_return(self, target_return: float,
                                allow_short: bool = False) -> Dict:
        """
        Find minimum variance portfolio for a target return.

        Args:
            target_return: Target annualized return
            allow_short: Whether to allow short selling

        Returns:
            Dictionary with optimal weights and portfolio metrics
        """
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - target_return}
        ]

        if allow_short:
            bounds = tuple((-1, 1) for _ in range(self.n_assets))
        else:
            bounds = tuple((0, 1) for _ in range(self.n_assets))

        def variance(weights):
            return self.portfolio_volatility(weights) ** 2

        result = minimize(
            variance,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-10}
        )

        if not result.success:
            print(f"Warning: Optimization may not have converged: {result.message}")

        optimal_weights = result.x
        optimal_weights[np.abs(optimal_weights) < 1e-6] = 0
        optimal_weights = optimal_weights / np.sum(optimal_weights)

        return self._create_result_dict(optimal_weights)

    def _create_result_dict(self, weights: np.ndarray) -> Dict:
        """Create a result dictionary with weights and metrics."""
        return {
            'weights': dict(zip(self.tickers, weights.tolist())),
            'expected_return': float(self.portfolio_return(weights)),
            'volatility': float(self.portfolio_volatility(weights)),
            'sharpe_ratio': float(self.portfolio_sharpe_ratio(weights))
        }

    def generate_efficient_frontier(self, n_points: int = 50,
                                     allow_short: bool = False) -> pd.DataFrame:
        """
        Generate points along the efficient frontier.

        Args:
            n_points: Number of points to generate
            allow_short: Whether to allow short selling

        Returns:
            DataFrame with return, volatility, and Sharpe ratio for each point
        """
        # Find min variance portfolio
        min_var = self.optimize_min_variance(allow_short)
        min_return = min_var['expected_return']

        # Find max return (100% in highest returning asset if no shorting)
        if allow_short:
            max_return = self.mean_returns.max() * 1.2
        else:
            max_return = self.mean_returns.max()

        target_returns = np.linspace(min_return, max_return, n_points)

        results = []
        for target in target_returns:
            try:
                result = self.optimize_target_return(target, allow_short)
                results.append({
                    'target_return': target,
                    'expected_return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe_ratio': result['sharpe_ratio']
                })
            except Exception:
                continue

        return pd.DataFrame(results)


def backtest_portfolio(weights: Dict[str, float], returns: pd.DataFrame,
                       initial_value: float = 100.0) -> pd.DataFrame:
    """
    Backtest a portfolio with given weights on historical returns.

    Args:
        weights: Dictionary mapping ticker to weight
        returns: DataFrame of asset returns (log or simple)
        initial_value: Initial portfolio value

    Returns:
        DataFrame with portfolio value over time
    """
    # Align weights with returns columns
    aligned_weights = []
    for col in returns.columns:
        aligned_weights.append(weights.get(col, 0))
    aligned_weights = np.array(aligned_weights)

    # Calculate portfolio returns
    portfolio_returns = (returns * aligned_weights).sum(axis=1)

    # Calculate cumulative returns and portfolio value
    cumulative_returns = (1 + portfolio_returns).cumprod()
    portfolio_value = initial_value * cumulative_returns

    result = pd.DataFrame({
        'daily_return': portfolio_returns,
        'cumulative_return': cumulative_returns,
        'portfolio_value': portfolio_value
    })

    return result


def calculate_backtest_metrics(backtest_results: pd.DataFrame,
                                risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate performance metrics from backtest results.

    Args:
        backtest_results: DataFrame from backtest_portfolio()
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary of performance metrics
    """
    daily_returns = backtest_results['daily_return']

    # Annualization factor
    ann_factor = 252

    total_return = backtest_results['cumulative_return'].iloc[-1] - 1
    ann_return = (1 + total_return) ** (ann_factor / len(daily_returns)) - 1
    ann_volatility = daily_returns.std() * np.sqrt(ann_factor)
    sharpe = (ann_return - risk_free_rate) / ann_volatility if ann_volatility > 0 else 0

    # Maximum drawdown
    cumulative = backtest_results['cumulative_return']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Calmar ratio
    calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'total_return': float(total_return),
        'annualized_return': float(ann_return),
        'annualized_volatility': float(ann_volatility),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar),
        'start_date': str(backtest_results.index[0].date()),
        'end_date': str(backtest_results.index[-1].date()),
        'trading_days': len(daily_returns)
    }


if __name__ == "__main__":
    # Test with sample data
    from data_loader import create_default_loader

    loader = create_default_loader()
    loader.download_data()
    loader.clean_data_pipeline()
    returns = loader.calculate_returns(log_returns=False)  # Use simple returns for backtest
    mean_ret, cov = loader.get_statistics()

    optimizer = PortfolioOptimizer(mean_ret, cov)

    print("\n--- Maximum Sharpe Ratio Portfolio ---")
    max_sharpe = optimizer.optimize_max_sharpe(allow_short=False)
    print(f"Expected Return: {max_sharpe['expected_return']:.2%}")
    print(f"Volatility: {max_sharpe['volatility']:.2%}")
    print(f"Sharpe Ratio: {max_sharpe['sharpe_ratio']:.2f}")
    print("\nWeights:")
    for ticker, weight in sorted(max_sharpe['weights'].items(), key=lambda x: -x[1]):
        if weight > 0.01:
            print(f"  {ticker}: {weight:.2%}")

    print("\n--- Backtest ---")
    backtest = backtest_portfolio(max_sharpe['weights'], returns)
    metrics = calculate_backtest_metrics(backtest)
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
