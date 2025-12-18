"""
Visualization Module
Creates charts and plots from portfolio analysis results.
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


def load_results(json_path: str) -> Dict:
    """Load portfolio analysis results from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def plot_portfolio_weights(results: Dict, output_path: str) -> None:
    """Create a bar chart of optimal portfolio weights."""
    weights = results['optimization']['max_sharpe_portfolio']['weights']

    # Filter out zero weights and sort by weight
    non_zero_weights = {k: v for k, v in weights.items() if v > 0.001}
    sorted_weights = dict(sorted(non_zero_weights.items(), key=lambda x: -x[1]))

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_weights)))
    bars = ax.bar(sorted_weights.keys(), sorted_weights.values(), color=colors, edgecolor='black')

    # Add value labels on bars
    for bar, val in zip(bars, sorted_weights.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Weight', fontsize=12)
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_title('Optimal Portfolio Weights (Maximum Sharpe Ratio)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(sorted_weights.values()) * 1.15)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_historical_performance(results: Dict, output_path: str) -> None:
    """Create a line chart of historical portfolio performance."""
    backtest = results['historical_backtest']
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in backtest['dates']]
    values = backtest['portfolio_values']

    fig, ax = plt.subplots(figsize=(14, 7))

    ax.plot(dates, values, color='#2E86AB', linewidth=1.5, label='Portfolio Value')
    ax.fill_between(dates, values, alpha=0.3, color='#2E86AB')

    # Add metrics annotation
    metrics = backtest['metrics']
    textstr = '\n'.join([
        f"Total Return: {metrics['total_return']:.1%}",
        f"Ann. Return: {metrics['annualized_return']:.1%}",
        f"Ann. Volatility: {metrics['annualized_volatility']:.1%}",
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}",
        f"Max Drawdown: {metrics['max_drawdown']:.1%}"
    ])

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Historical Portfolio Performance (Backtest)', fontsize=14, fontweight='bold')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_drawdown(results: Dict, output_path: str) -> None:
    """Create a drawdown chart."""
    backtest = results['historical_backtest']
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in backtest['dates']]
    cumulative = np.array(backtest['cumulative_returns'])

    # Calculate drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100  # Convert to percentage

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.fill_between(dates, drawdown, 0, color='#E74C3C', alpha=0.7)
    ax.plot(dates, drawdown, color='#C0392B', linewidth=0.8)

    # Mark maximum drawdown
    min_idx = np.argmin(drawdown)
    ax.scatter([dates[min_idx]], [drawdown[min_idx]], color='darkred', s=100, zorder=5)
    ax.annotate(f'Max DD: {drawdown[min_idx]:.1f}%',
                xy=(dates[min_idx], drawdown[min_idx]),
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, fontweight='bold', color='darkred')

    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Portfolio Drawdown', fontsize=14, fontweight='bold')

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min(drawdown) * 1.1, 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_stress_simulation(results: Dict, output_path: str) -> None:
    """Create a chart showing stress scenario simulation paths."""
    stress = results['stress_scenario']
    forward = stress['forward_simulation']

    dates = [datetime.strptime(d, '%Y-%m-%d') for d in forward['dates']]
    avg_path = forward['avg_portfolio_value']
    lower_band = forward['lower_band_5pct']
    upper_band = forward['upper_band_95pct']

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot confidence bands
    ax.fill_between(dates, lower_band, upper_band, alpha=0.3, color='#E74C3C', label='5th-95th Percentile')
    ax.plot(dates, avg_path, color='#2E86AB', linewidth=2, label='Mean Path')
    ax.plot(dates, lower_band, color='#E74C3C', linewidth=1, linestyle='--', alpha=0.7)
    ax.plot(dates, upper_band, color='#27AE60', linewidth=1, linestyle='--', alpha=0.7)

    # Add horizontal line at 1 (starting value)
    ax.axhline(y=1, color='gray', linestyle=':', linewidth=1, alpha=0.7)

    # Add stress scenario info
    config = stress['config']
    sim_results = stress['simulation_results']
    textstr = '\n'.join([
        f"Scenario: {config['name']}",
        f"Drop: {config['expected_drop_pct']:.0%} ± {config['drop_uncertainty_pct']:.0%}",
        f"Vol Increase: {config['volatility_increase_pct']:.0%}",
        f"",
        f"Mean Return: {sim_results['mean_return']:.1%}",
        f"95% VaR: {sim_results['var_95']:.1%}",
        f"ES (95%): {sim_results['expected_shortfall_95']:.1%}"
    ])

    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')

    ax.set_ylabel('Portfolio Value (Relative)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title('Forward Stress Scenario Simulation', fontsize=14, fontweight='bold')

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_return_distribution(results: Dict, output_path: str) -> None:
    """Create a histogram of simulated returns."""
    stress = results['stress_scenario']
    returns = np.array(stress['forward_simulation']['return_distribution']) * 100  # Convert to %
    sim_results = stress['simulation_results']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create histogram
    n, bins, patches = ax.hist(returns, bins=50, density=True, alpha=0.7,
                                color='#3498DB', edgecolor='black', linewidth=0.5)

    # Color bars below VaR in red
    var_95 = sim_results['var_95'] * 100
    for patch, left_edge in zip(patches, bins[:-1]):
        if left_edge < var_95:
            patch.set_facecolor('#E74C3C')

    # Add vertical lines for key statistics
    ax.axvline(x=np.mean(returns), color='#2E86AB', linewidth=2, linestyle='-', label=f'Mean: {np.mean(returns):.1f}%')
    ax.axvline(x=np.median(returns), color='#27AE60', linewidth=2, linestyle='--', label=f'Median: {np.median(returns):.1f}%')
    ax.axvline(x=var_95, color='#E74C3C', linewidth=2, linestyle=':', label=f'95% VaR: {var_95:.1f}%')

    ax.set_xlabel('Return (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Simulated Returns (Stress Scenario)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_risk_metrics_comparison(results: Dict, output_path: str) -> None:
    """Create a comparison chart of risk metrics."""
    backtest_metrics = results['historical_backtest']['metrics']
    stress_metrics = results['stress_scenario']['simulation_results']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Returns comparison
    ax1 = axes[0]
    categories = ['Historical\n(Annualized)', 'Stress\n(Mean)', 'Stress\n(Median)']
    values = [
        backtest_metrics['annualized_return'] * 100,
        stress_metrics['mean_return'] * 100,
        stress_metrics['median_return'] * 100
    ]
    colors = ['#2E86AB', '#E74C3C', '#E74C3C']
    bars = ax1.bar(categories, values, color=colors, edgecolor='black')
    ax1.set_ylabel('Return (%)', fontsize=11)
    ax1.set_title('Expected Returns', fontsize=12, fontweight='bold')
    ax1.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Chart 2: Risk metrics
    ax2 = axes[1]
    categories = ['Historical\nVolatility', 'Stress\nStd Dev', 'Historical\nMax DD']
    values = [
        backtest_metrics['annualized_volatility'] * 100,
        stress_metrics['std_return'] * 100,
        abs(backtest_metrics['max_drawdown']) * 100
    ]
    colors = ['#F39C12', '#E74C3C', '#9B59B6']
    bars = ax2.bar(categories, values, color=colors, edgecolor='black')
    ax2.set_ylabel('Risk (%)', fontsize=11)
    ax2.set_title('Risk Measures', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Chart 3: Tail risk
    ax3 = axes[2]
    categories = ['95% VaR', 'Expected\nShortfall', 'Mean Max\nDrawdown']
    values = [
        abs(stress_metrics['var_95']) * 100,
        abs(stress_metrics['expected_shortfall_95']) * 100,
        abs(stress_metrics['mean_max_drawdown']) * 100
    ]
    colors = ['#E74C3C', '#C0392B', '#922B21']
    bars = ax3.bar(categories, values, color=colors, edgecolor='black')
    ax3.set_ylabel('Loss (%)', fontsize=11)
    ax3.set_title('Tail Risk (Stress Scenario)', fontsize=12, fontweight='bold')
    for bar, val in zip(bars, values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle('Portfolio Risk Metrics Summary', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def create_summary_dashboard(results: Dict, output_path: str) -> None:
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 12))

    # Grid layout
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Portfolio Weights (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    weights = results['optimization']['max_sharpe_portfolio']['weights']
    non_zero = {k: v for k, v in weights.items() if v > 0.001}
    sorted_w = dict(sorted(non_zero.items(), key=lambda x: -x[1]))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_w)))
    ax1.barh(list(sorted_w.keys())[::-1], list(sorted_w.values())[::-1], color=colors[::-1])
    ax1.set_xlabel('Weight')
    ax1.set_title('Portfolio Weights', fontweight='bold')
    ax1.set_xlim(0, max(sorted_w.values()) * 1.2)

    # 2. Historical Performance (top middle and right)
    ax2 = fig.add_subplot(gs[0, 1:])
    backtest = results['historical_backtest']
    dates = [datetime.strptime(d, '%Y-%m-%d') for d in backtest['dates']]
    ax2.plot(dates, backtest['portfolio_values'], color='#2E86AB', linewidth=1.5)
    ax2.fill_between(dates, backtest['portfolio_values'], alpha=0.3, color='#2E86AB')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('Historical Performance', fontweight='bold')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.grid(True, alpha=0.3)

    # 3. Drawdown (middle left and center)
    ax3 = fig.add_subplot(gs[1, :2])
    cumulative = np.array(backtest['cumulative_returns'])
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max * 100
    ax3.fill_between(dates, drawdown, 0, color='#E74C3C', alpha=0.7)
    ax3.set_ylabel('Drawdown (%)')
    ax3.set_title('Drawdown Analysis', fontweight='bold')
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax3.grid(True, alpha=0.3)

    # 4. Metrics Table (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')
    metrics = backtest['metrics']
    stress = results['stress_scenario']['simulation_results']
    table_data = [
        ['Metric', 'Historical', 'Stress'],
        ['Return', f"{metrics['annualized_return']:.1%}", f"{stress['mean_return']:.1%}"],
        ['Volatility', f"{metrics['annualized_volatility']:.1%}", f"{stress['std_return']:.1%}"],
        ['Sharpe', f"{metrics['sharpe_ratio']:.2f}", '-'],
        ['Max DD', f"{metrics['max_drawdown']:.1%}", f"{stress['mean_max_drawdown']:.1%}"],
        ['VaR (95%)', '-', f"{stress['var_95']:.1%}"],
        ['ES (95%)', '-', f"{stress['expected_shortfall_95']:.1%}"],
    ]
    table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for i in range(3):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    ax4.set_title('Key Metrics', fontweight='bold', pad=20)

    # 5. Stress Simulation (bottom left and center)
    ax5 = fig.add_subplot(gs[2, :2])
    forward = results['stress_scenario']['forward_simulation']
    stress_dates = [datetime.strptime(d, '%Y-%m-%d') for d in forward['dates']]
    ax5.fill_between(stress_dates, forward['lower_band_5pct'], forward['upper_band_95pct'],
                     alpha=0.3, color='#E74C3C', label='5th-95th %ile')
    ax5.plot(stress_dates, forward['avg_portfolio_value'], color='#2E86AB', linewidth=2, label='Mean')
    ax5.axhline(y=1, color='gray', linestyle=':', linewidth=1)
    ax5.set_ylabel('Portfolio Value')
    ax5.set_title('Stress Scenario Forward Simulation', fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax5.grid(True, alpha=0.3)
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 6. Return Distribution (bottom right)
    ax6 = fig.add_subplot(gs[2, 2])
    returns = np.array(forward['return_distribution']) * 100
    ax6.hist(returns, bins=30, density=True, alpha=0.7, color='#3498DB', edgecolor='black')
    ax6.axvline(x=np.mean(returns), color='#2E86AB', linewidth=2, label=f'Mean: {np.mean(returns):.1f}%')
    ax6.axvline(x=stress['var_95']*100, color='#E74C3C', linewidth=2, linestyle=':', label=f'VaR: {stress["var_95"]*100:.1f}%')
    ax6.set_xlabel('Return (%)')
    ax6.set_ylabel('Density')
    ax6.set_title('Return Distribution', fontweight='bold')
    ax6.legend(loc='upper right', fontsize=8)

    # Main title
    plt.suptitle('Portfolio Optimization & Stress Testing Dashboard',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_all_visualizations(json_path: str, output_dir: str) -> None:
    """Generate all visualizations from results JSON."""
    print("=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)

    # Load results
    results = load_results(json_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nInput: {json_path}")
    print(f"Output directory: {output_dir}\n")

    # Generate individual plots
    plot_portfolio_weights(results, str(output_path / '1_portfolio_weights.png'))
    plot_historical_performance(results, str(output_path / '2_historical_performance.png'))
    plot_drawdown(results, str(output_path / '3_drawdown.png'))
    plot_stress_simulation(results, str(output_path / '4_stress_simulation.png'))
    plot_return_distribution(results, str(output_path / '5_return_distribution.png'))
    plot_risk_metrics_comparison(results, str(output_path / '6_risk_metrics.png'))

    # Generate summary dashboard
    create_summary_dashboard(results, str(output_path / '0_dashboard.png'))

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate portfolio analysis visualizations')
    parser.add_argument('--input', '-i', type=str,
                        default='result/portfolio_analysis_results.json',
                        help='Input JSON file path')
    parser.add_argument('--output', '-o', type=str,
                        default='result',
                        help='Output directory for visualizations')

    args = parser.parse_args()
    generate_all_visualizations(args.input, args.output)


if __name__ == "__main__":
    main()
