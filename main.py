"""
Main Module
Orchestrates portfolio optimization and stress testing, generates JSON output.
"""

import json
import argparse
from datetime import datetime
from typing import Dict, Optional

from data_loader import DataLoader, create_default_loader
from portfolio_optimizer import (
    PortfolioOptimizer,
    backtest_portfolio,
    calculate_backtest_metrics
)
from scenario_model import (
    ScenarioSimulator,
    create_trump_tariff_scenario,
    create_custom_scenario,
    StressScenarioConfig
)


def run_portfolio_analysis(
    tickers: Optional[list] = None,
    years_of_history: int = 10,
    forward_horizon_days: int = 252,
    scenario_config: Optional[StressScenarioConfig] = None,
    allow_short: bool = False,
    risk_free_rate: float = 0.02,
    n_simulations: int = 1000,
    output_file: Optional[str] = None,
    seed: Optional[int] = None
) -> Dict:
    """
    Run the complete portfolio analysis pipeline.

    Args:
        tickers: List of tickers to analyze (uses default if None)
        years_of_history: Years of historical data to use
        forward_horizon_days: Number of trading days for forward simulation
        scenario_config: Stress scenario configuration (uses Trump tariff default if None)
        allow_short: Whether to allow short selling
        risk_free_rate: Risk-free rate for Sharpe ratio
        n_simulations: Number of Monte Carlo simulations
        output_file: Path to save JSON output (optional)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with complete analysis results
    """
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION WITH STRESS TESTING")
    print("=" * 60)

    # Step 1: Load and clean data
    print("\n[1/5] Loading and cleaning data...")
    if tickers:
        loader = DataLoader(tickers, years_of_history)
    else:
        loader = create_default_loader()

    loader.download_data()
    loader.clean_data_pipeline()

    # Use simple returns for backtesting
    returns = loader.calculate_returns(log_returns=False)
    mean_returns, cov_matrix = loader.get_statistics()
    available_tickers = loader.get_available_tickers()

    print(f"Available tickers after cleaning: {len(available_tickers)}")

    # Step 2: Optimize portfolio
    print("\n[2/5] Running portfolio optimization...")
    optimizer = PortfolioOptimizer(mean_returns, cov_matrix, risk_free_rate)

    # Get max Sharpe portfolio
    max_sharpe_result = optimizer.optimize_max_sharpe(allow_short=allow_short)

    # Get min variance portfolio
    min_var_result = optimizer.optimize_min_variance(allow_short=allow_short)

    print(f"Max Sharpe Portfolio:")
    print(f"  Expected Return: {max_sharpe_result['expected_return']:.2%}")
    print(f"  Volatility: {max_sharpe_result['volatility']:.2%}")
    print(f"  Sharpe Ratio: {max_sharpe_result['sharpe_ratio']:.2f}")

    # Step 3: Backtest on historical data
    print("\n[3/5] Running historical backtest...")
    backtest_result = backtest_portfolio(max_sharpe_result['weights'], returns)
    backtest_metrics = calculate_backtest_metrics(backtest_result, risk_free_rate)

    print(f"Backtest Results ({backtest_metrics['start_date']} to {backtest_metrics['end_date']}):")
    print(f"  Total Return: {backtest_metrics['total_return']:.2%}")
    print(f"  Annualized Return: {backtest_metrics['annualized_return']:.2%}")
    print(f"  Max Drawdown: {backtest_metrics['max_drawdown']:.2%}")
    print(f"  Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.2f}")

    # Step 4: Create stress scenario
    print("\n[4/5] Setting up stress scenario...")
    if scenario_config is None:
        scenario_config = create_trump_tariff_scenario(
            duration_days=forward_horizon_days,
            expected_drop=0.10,
            drop_uncertainty=0.02,
            volatility_increase=0.10,
            n_simulations=n_simulations
        )

    print(f"Scenario: {scenario_config.name}")
    print(f"  Duration: {scenario_config.duration_days} trading days")
    print(f"  Expected Drop: {scenario_config.expected_drop_pct:.1%} ± {scenario_config.drop_uncertainty_pct:.1%}")
    print(f"  Volatility Increase: {scenario_config.volatility_increase_pct:.1%}")

    # Count affected assets that are in our portfolio
    affected_in_portfolio = [
        t for t in scenario_config.affected_assets
        if t in available_tickers
    ]
    print(f"  Affected assets in portfolio: {len(affected_in_portfolio)}")

    # Step 5: Run forward simulation
    print("\n[5/5] Running forward stress simulation...")
    last_prices = loader.clean_data.iloc[-1]
    simulator = ScenarioSimulator(mean_returns, cov_matrix, last_prices)

    sim_result = simulator.simulate_gbm(scenario_config, seed=seed)
    stress_performance = simulator.calculate_portfolio_performance(
        sim_result, max_sharpe_result['weights']
    )

    print(f"Stress Scenario Results ({n_simulations} simulations):")
    print(f"  Mean Return: {stress_performance['mean_return']:.2%}")
    print(f"  Median Return: {stress_performance['median_return']:.2%}")
    print(f"  Std Return: {stress_performance['std_return']:.2%}")
    print(f"  95% VaR: {stress_performance['var_95']:.2%}")
    print(f"  Expected Shortfall (95%): {stress_performance['expected_shortfall_95']:.2%}")

    # Compile results
    results = {
        'metadata': {
            'run_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tickers_requested': tickers if tickers else 'default',
            'tickers_available': available_tickers,
            'years_of_history': years_of_history,
            'forward_horizon_days': forward_horizon_days,
            'allow_short_selling': allow_short,
            'risk_free_rate': risk_free_rate,
            'n_simulations': n_simulations
        },
        'optimization': {
            'max_sharpe_portfolio': max_sharpe_result,
            'min_variance_portfolio': min_var_result
        },
        'historical_backtest': {
            'metrics': backtest_metrics,
            'daily_returns': backtest_result['daily_return'].tolist(),
            'cumulative_returns': backtest_result['cumulative_return'].tolist(),
            'portfolio_values': backtest_result['portfolio_value'].tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in backtest_result.index]
        },
        'stress_scenario': {
            'config': {
                'name': scenario_config.name,
                'start_date': scenario_config.start_date.strftime('%Y-%m-%d'),
                'duration_days': scenario_config.duration_days,
                'affected_assets': scenario_config.affected_assets,
                'expected_drop_pct': scenario_config.expected_drop_pct,
                'drop_uncertainty_pct': scenario_config.drop_uncertainty_pct,
                'volatility_increase_pct': scenario_config.volatility_increase_pct
            },
            'simulation_results': {
                'mean_return': stress_performance['mean_return'],
                'std_return': stress_performance['std_return'],
                'median_return': stress_performance['median_return'],
                'percentiles': stress_performance['percentiles'],
                'var_95': stress_performance['var_95'],
                'var_99': stress_performance['var_99'],
                'expected_shortfall_95': stress_performance['expected_shortfall_95'],
                'mean_max_drawdown': stress_performance['mean_max_drawdown']
            },
            'forward_simulation': {
                'dates': stress_performance['avg_path_dates'],
                'avg_portfolio_value': stress_performance['avg_path'],
                'lower_band_5pct': stress_performance['lower_band_5pct'],
                'upper_band_95pct': stress_performance['upper_band_95pct'],
                'return_distribution': stress_performance['return_distribution']
            }
        }
    }

    # Save to file if specified
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    return results


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Portfolio Optimization with Stress Testing'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='portfolio_analysis_results.json',
        help='Output JSON file path'
    )

    parser.add_argument(
        '--years',
        type=int,
        default=10,
        help='Years of historical data (default: 10)'
    )

    parser.add_argument(
        '--horizon',
        type=int,
        default=252,
        help='Forward simulation horizon in trading days (default: 252 = 1 year)'
    )

    parser.add_argument(
        '--simulations', '-n',
        type=int,
        default=1000,
        help='Number of Monte Carlo simulations (default: 1000)'
    )

    parser.add_argument(
        '--drop',
        type=float,
        default=0.10,
        help='Expected drop percentage for stressed assets (default: 0.10)'
    )

    parser.add_argument(
        '--drop-uncertainty',
        type=float,
        default=0.02,
        help='Uncertainty around the drop (default: 0.02)'
    )

    parser.add_argument(
        '--vol-increase',
        type=float,
        default=0.10,
        help='Volatility increase percentage (default: 0.10)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Create custom scenario with CLI arguments
    scenario = create_trump_tariff_scenario(
        duration_days=args.horizon,
        expected_drop=args.drop,
        drop_uncertainty=args.drop_uncertainty,
        volatility_increase=args.vol_increase,
        n_simulations=args.simulations
    )

    # Run analysis
    results = run_portfolio_analysis(
        years_of_history=args.years,
        forward_horizon_days=args.horizon,
        scenario_config=scenario,
        n_simulations=args.simulations,
        output_file=args.output,
        seed=args.seed
    )

    return results


if __name__ == "__main__":
    main()
