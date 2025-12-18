"""
Scenario Modeling Module
Implements stress testing scenarios using Geometric Brownian Motion (GBM).
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StressScenarioConfig:
    """Configuration for a stress scenario."""
    name: str
    start_date: datetime
    duration_days: int
    affected_assets: List[str]
    expected_drop_pct: float  # e.g., 0.10 for 10% drop
    drop_uncertainty_pct: float  # e.g., 0.02 for ±2% uncertainty
    volatility_increase_pct: float  # e.g., 0.10 for 10% increase in volatility
    n_simulations: int = 1000


def create_trump_tariff_scenario(
    start_date: Optional[datetime] = None,
    duration_days: int = 252,
    expected_drop: float = 0.10,
    drop_uncertainty: float = 0.02,
    volatility_increase: float = 0.10,
    n_simulations: int = 1000
) -> StressScenarioConfig:
    """
    Create a Trump tariff stress scenario configuration.

    Args:
        start_date: Start date of the scenario (defaults to today)
        duration_days: Duration of forward simulation in trading days
        expected_drop: Expected percentage drop for affected assets (e.g., 0.10 = 10%)
        drop_uncertainty: Uncertainty around the drop (e.g., 0.02 = ±2%)
        volatility_increase: Percentage increase in volatility (e.g., 0.10 = 10%)
        n_simulations: Number of Monte Carlo simulations

    Returns:
        StressScenarioConfig for the scenario
    """
    if start_date is None:
        start_date = datetime.now()

    # Thai-related assets affected by Trump tariffs
    thai_assets = [
        # Thai Export companies
        'DELTA.BK', 'HANA.BK', 'STA.BK', 'IVL.BK', 'PTTGC.BK',
        # Thai Domestic companies
        'CPALL.BK', 'AOT.BK', 'BDMS.BK', 'SCB.BK', 'CPN.BK', 'MINT.BK',
        # Thai-related ETFs and currency
        'THD', 'THB=X'
    ]

    return StressScenarioConfig(
        name="Trump Tariff Scenario",
        start_date=start_date,
        duration_days=duration_days,
        affected_assets=thai_assets,
        expected_drop_pct=expected_drop,
        drop_uncertainty_pct=drop_uncertainty,
        volatility_increase_pct=volatility_increase,
        n_simulations=n_simulations
    )


class ScenarioSimulator:
    """Simulates portfolio performance under stress scenarios using GBM."""

    def __init__(self, mean_returns: pd.Series, cov_matrix: pd.DataFrame,
                 last_prices: pd.Series):
        """
        Initialize the simulator.

        Args:
            mean_returns: Annualized mean returns for each asset
            cov_matrix: Annualized covariance matrix
            last_prices: Last observed prices for each asset
        """
        self.mean_returns = mean_returns
        self.cov_matrix = cov_matrix
        self.last_prices = last_prices
        self.tickers = list(mean_returns.index)

    def simulate_gbm(self, config: StressScenarioConfig,
                     seed: Optional[int] = None) -> Dict:
        """
        Run GBM simulation under stress scenario.

        Args:
            config: Stress scenario configuration
            seed: Random seed for reproducibility

        Returns:
            Dictionary with simulation results
        """
        if seed is not None:
            np.random.seed(seed)

        n_assets = len(self.tickers)
        n_days = config.duration_days
        n_sims = config.n_simulations
        dt = 1 / 252  # Daily time step

        # Prepare adjusted parameters for stressed assets
        stressed_mu = self.mean_returns.copy()
        stressed_cov = self.cov_matrix.copy()

        for ticker in config.affected_assets:
            if ticker in self.tickers:
                idx = self.tickers.index(ticker)

                # Adjust drift to reflect expected drop
                # The drop happens over the simulation period
                # Sample from uniform distribution for the drop
                drop = config.expected_drop_pct + np.random.uniform(
                    -config.drop_uncertainty_pct,
                    config.drop_uncertainty_pct,
                    n_sims
                )

                # Adjust volatility (increase by specified percentage)
                original_vol = np.sqrt(stressed_cov.iloc[idx, idx])
                new_vol = original_vol * (1 + config.volatility_increase_pct)

                # Update covariance matrix for this asset
                # Scale the row and column by the volatility increase ratio
                vol_ratio = new_vol / original_vol if original_vol > 0 else 1
                stressed_cov.iloc[idx, :] *= vol_ratio
                stressed_cov.iloc[:, idx] *= vol_ratio

        # Extract parameters
        mu = stressed_mu.values
        sigma = np.sqrt(np.diag(stressed_cov.values))  # Asset volatilities

        # Cholesky decomposition for correlated random draws
        try:
            L = np.linalg.cholesky(stressed_cov.values)
        except np.linalg.LinAlgError:
            # If not positive definite, use eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eigh(stressed_cov.values)
            eigenvalues = np.maximum(eigenvalues, 1e-8)
            L = eigenvectors @ np.diag(np.sqrt(eigenvalues))

        # Initialize price paths
        # Shape: (n_sims, n_days + 1, n_assets)
        price_paths = np.zeros((n_sims, n_days + 1, n_assets))
        price_paths[:, 0, :] = self.last_prices.values

        # Apply initial shock to affected assets
        for i, ticker in enumerate(self.tickers):
            if ticker in config.affected_assets:
                # Sample the drop for each simulation
                drops = config.expected_drop_pct + np.random.uniform(
                    -config.drop_uncertainty_pct,
                    config.drop_uncertainty_pct,
                    n_sims
                )
                price_paths[:, 0, i] *= (1 - drops)

        # Simulate GBM paths
        for t in range(1, n_days + 1):
            # Generate correlated random numbers
            Z = np.random.standard_normal((n_sims, n_assets))
            correlated_Z = Z @ L.T

            # GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * np.sqrt(dt) * correlated_Z

            price_paths[:, t, :] = price_paths[:, t - 1, :] * np.exp(drift + diffusion)

        return {
            'price_paths': price_paths,
            'tickers': self.tickers,
            'config': config
        }

    def calculate_portfolio_performance(self, simulation_result: Dict,
                                         weights: Dict[str, float]) -> Dict:
        """
        Calculate portfolio performance from simulation results.

        Args:
            simulation_result: Output from simulate_gbm()
            weights: Portfolio weights as dictionary

        Returns:
            Dictionary with portfolio performance metrics
        """
        price_paths = simulation_result['price_paths']
        tickers = simulation_result['tickers']

        n_sims, n_days_plus_1, n_assets = price_paths.shape
        n_days = n_days_plus_1 - 1

        # Align weights with tickers
        aligned_weights = np.array([weights.get(t, 0) for t in tickers])

        # Calculate portfolio value paths
        # Portfolio value = sum(price * weight / initial_price)
        initial_prices = price_paths[:, 0, :]
        portfolio_paths = np.zeros((n_sims, n_days_plus_1))

        for t in range(n_days_plus_1):
            # Relative price change for each asset
            relative_prices = price_paths[:, t, :] / initial_prices
            # Portfolio value (assuming weights sum to 1 and initial value = 1)
            portfolio_paths[:, t] = np.sum(relative_prices * aligned_weights, axis=1)

        # Calculate returns
        final_values = portfolio_paths[:, -1]
        total_returns = final_values - 1  # Since initial value = 1

        # Calculate statistics across simulations
        mean_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        median_return = np.median(total_returns)

        # Percentiles
        percentiles = {
            'p5': np.percentile(total_returns, 5),
            'p25': np.percentile(total_returns, 25),
            'p50': np.percentile(total_returns, 50),
            'p75': np.percentile(total_returns, 75),
            'p95': np.percentile(total_returns, 95)
        }

        # Value at Risk and Expected Shortfall
        var_95 = np.percentile(total_returns, 5)  # 95% VaR
        var_99 = np.percentile(total_returns, 1)  # 99% VaR
        es_95 = np.mean(total_returns[total_returns <= var_95])  # Expected Shortfall

        # Maximum drawdown for each simulation
        max_drawdowns = []
        for sim in range(n_sims):
            path = portfolio_paths[sim, :]
            running_max = np.maximum.accumulate(path)
            drawdown = (path - running_max) / running_max
            max_drawdowns.append(np.min(drawdown))

        mean_max_drawdown = np.mean(max_drawdowns)

        # Generate time series for average path
        dates = pd.date_range(
            start=simulation_result['config'].start_date,
            periods=n_days_plus_1,
            freq='B'  # Business days
        )

        avg_path = pd.Series(
            np.mean(portfolio_paths, axis=0),
            index=dates,
            name='avg_portfolio_value'
        )

        # Also get percentile bands
        lower_band = pd.Series(
            np.percentile(portfolio_paths, 5, axis=0),
            index=dates,
            name='p5_portfolio_value'
        )
        upper_band = pd.Series(
            np.percentile(portfolio_paths, 95, axis=0),
            index=dates,
            name='p95_portfolio_value'
        )

        return {
            'scenario_name': simulation_result['config'].name,
            'n_simulations': n_sims,
            'horizon_days': n_days,
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'median_return': float(median_return),
            'percentiles': percentiles,
            'var_95': float(var_95),
            'var_99': float(var_99),
            'expected_shortfall_95': float(es_95),
            'mean_max_drawdown': float(mean_max_drawdown),
            'return_distribution': total_returns.tolist(),
            'avg_path': avg_path.tolist(),
            'avg_path_dates': [d.strftime('%Y-%m-%d') for d in dates],
            'lower_band_5pct': lower_band.tolist(),
            'upper_band_95pct': upper_band.tolist()
        }


def create_custom_scenario(
    name: str,
    affected_assets: List[str],
    start_date: Optional[datetime] = None,
    duration_days: int = 252,
    expected_drop: float = 0.10,
    drop_uncertainty: float = 0.02,
    volatility_increase: float = 0.10,
    n_simulations: int = 1000
) -> StressScenarioConfig:
    """
    Create a custom stress scenario configuration.

    Args:
        name: Name of the scenario
        affected_assets: List of tickers affected by the stress
        start_date: Start date of the scenario
        duration_days: Duration in trading days
        expected_drop: Expected percentage drop for affected assets
        drop_uncertainty: Uncertainty around the drop (±)
        volatility_increase: Percentage increase in volatility
        n_simulations: Number of Monte Carlo simulations

    Returns:
        StressScenarioConfig for the scenario
    """
    if start_date is None:
        start_date = datetime.now()

    return StressScenarioConfig(
        name=name,
        start_date=start_date,
        duration_days=duration_days,
        affected_assets=affected_assets,
        expected_drop_pct=expected_drop,
        drop_uncertainty_pct=drop_uncertainty,
        volatility_increase_pct=volatility_increase,
        n_simulations=n_simulations
    )


if __name__ == "__main__":
    # Test the scenario simulator
    from data_loader import create_default_loader

    loader = create_default_loader()
    loader.download_data()
    loader.clean_data_pipeline()
    loader.calculate_returns()
    mean_ret, cov = loader.get_statistics()
    last_prices = loader.clean_data.iloc[-1]

    simulator = ScenarioSimulator(mean_ret, cov, last_prices)

    # Create Trump tariff scenario
    scenario = create_trump_tariff_scenario(
        expected_drop=0.10,
        drop_uncertainty=0.02,
        volatility_increase=0.10,
        n_simulations=100
    )

    print(f"\n--- {scenario.name} ---")
    print(f"Duration: {scenario.duration_days} trading days")
    print(f"Expected Drop: {scenario.expected_drop_pct:.1%} ± {scenario.drop_uncertainty_pct:.1%}")
    print(f"Volatility Increase: {scenario.volatility_increase_pct:.1%}")
    print(f"Affected Assets: {len(scenario.affected_assets)} assets")

    # Run simulation with equal weights
    weights = {ticker: 1 / len(mean_ret) for ticker in mean_ret.index}

    sim_result = simulator.simulate_gbm(scenario, seed=42)
    perf = simulator.calculate_portfolio_performance(sim_result, weights)

    print(f"\nSimulation Results (Equal Weight Portfolio):")
    print(f"Mean Return: {perf['mean_return']:.2%}")
    print(f"Std Return: {perf['std_return']:.2%}")
    print(f"95% VaR: {perf['var_95']:.2%}")
    print(f"Expected Shortfall: {perf['expected_shortfall_95']:.2%}")
