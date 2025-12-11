---
name: backtesting-patterns
description: Load when user needs backtesting frameworks, walk-forward analysis, overfitting detection, transaction cost modeling, or strategy validation
trigger_keywords: [backtest, backtesting, walk forward, walk-forward, overfitting, out of sample, in sample, transaction cost, slippage, vectorbt, backtrader, zipline, strategy validation, paper trading, historical simulation, monte carlo, bootstrap]
---

# Backtesting Patterns Skill

Production-grade backtesting methodologies including walk-forward analysis, overfitting detection, transaction cost modeling, and robust strategy validation.

## Core Concepts

### Backtesting Hierarchy

```yaml
Level 1 - Vectorized Backtest:
  Speed: Very fast (millions of bars/second)
  Accuracy: Low (no market dynamics)
  Use: Initial idea screening

Level 2 - Event-Driven Backtest:
  Speed: Moderate (thousands of bars/second)
  Accuracy: Medium (simulates order flow)
  Use: Strategy development

Level 3 - Market Simulation:
  Speed: Slow (hundreds of bars/second)
  Accuracy: High (realistic execution)
  Use: Final validation

Level 4 - Paper Trading:
  Speed: Real-time
  Accuracy: Highest (live data)
  Use: Pre-production validation
```

### Common Biases to Avoid

```yaml
Look-Ahead Bias:
  Cause: Using future data in decisions
  Prevention: Strict time-series indexing, lag all features

Survivorship Bias:
  Cause: Only testing on stocks that still exist
  Prevention: Use point-in-time constituent data

Overfitting:
  Cause: Too many parameters, insufficient data
  Prevention: Walk-forward analysis, out-of-sample testing

Transaction Cost Bias:
  Cause: Ignoring realistic costs
  Prevention: Model spread, slippage, fees, market impact

Selection Bias:
  Cause: Testing many strategies, reporting best
  Prevention: Pre-registration, multiple testing correction
```

## Implementation Patterns

### 1. Vectorized Backtest Engine

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, Dict
from enum import Enum

class PositionSizing(Enum):
    FIXED_QUANTITY = "fixed_quantity"
    FIXED_VALUE = "fixed_value"
    PERCENT_EQUITY = "percent_equity"
    KELLY = "kelly"
    VOLATILITY_TARGET = "volatility_target"

@dataclass
class BacktestConfig:
    """Configuration for vectorized backtest"""
    initial_capital: float = 100_000
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    slippage_pct: float = 0.001  # 10 bps slippage
    position_sizing: PositionSizing = PositionSizing.PERCENT_EQUITY
    position_size_param: float = 0.1  # 10% of equity per position
    max_positions: int = 10
    allow_shorting: bool = False

@dataclass
class BacktestResult:
    """Backtest results container"""
    returns: pd.Series
    positions: pd.DataFrame
    trades: pd.DataFrame
    equity_curve: pd.Series
    metrics: Dict[str, float]

class VectorizedBacktest:
    """
    Fast vectorized backtesting engine

    Suitable for initial strategy screening
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        prices: pd.DataFrame,  # OHLCV with columns: open, high, low, close, volume
        signals: pd.Series,     # Signal: 1=long, -1=short, 0=flat
        entry_prices: Optional[pd.Series] = None  # Default: next open
    ) -> BacktestResult:
        """
        Run vectorized backtest

        Args:
            prices: OHLCV price data
            signals: Trading signals (-1, 0, 1)
            entry_prices: Execution prices (default: next bar open)

        Returns:
            BacktestResult with performance metrics
        """
        if entry_prices is None:
            entry_prices = prices['open'].shift(-1)  # Next bar open

        # Calculate position changes
        position_changes = signals.diff().fillna(signals)

        # Calculate returns
        strategy_returns = self._calculate_returns(
            prices, signals, entry_prices, position_changes
        )

        # Build equity curve
        equity_curve = self._build_equity_curve(strategy_returns)

        # Generate trade log
        trades = self._generate_trades(prices, signals, entry_prices)

        # Calculate metrics
        metrics = self._calculate_metrics(strategy_returns, equity_curve, trades)

        return BacktestResult(
            returns=strategy_returns,
            positions=signals.to_frame('position'),
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )

    def _calculate_returns(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        entry_prices: pd.Series,
        position_changes: pd.Series
    ) -> pd.Series:
        """Calculate strategy returns with costs"""
        # Price returns
        price_returns = prices['close'].pct_change()

        # Strategy returns (signal applied to returns)
        strategy_returns = signals.shift(1) * price_returns

        # Transaction costs
        trade_value = abs(position_changes) * entry_prices
        commission = np.maximum(
            trade_value * self.config.commission_per_share,
            self.config.commission_min * (abs(position_changes) > 0)
        )

        # Slippage
        slippage = abs(position_changes) * entry_prices * self.config.slippage_pct

        # Net returns
        total_costs = (commission + slippage) / self.config.initial_capital
        net_returns = strategy_returns - total_costs

        return net_returns

    def _build_equity_curve(self, returns: pd.Series) -> pd.Series:
        """Build equity curve from returns"""
        return self.config.initial_capital * (1 + returns).cumprod()

    def _generate_trades(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        entry_prices: pd.Series
    ) -> pd.DataFrame:
        """Generate trade log"""
        position_changes = signals.diff().fillna(signals)
        trade_mask = position_changes != 0

        trades = pd.DataFrame({
            'date': prices.index[trade_mask],
            'side': np.where(position_changes[trade_mask] > 0, 'buy', 'sell'),
            'price': entry_prices[trade_mask],
            'signal': signals[trade_mask]
        })

        return trades

    def _calculate_metrics(
        self,
        returns: pd.Series,
        equity_curve: pd.Series,
        trades: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics"""
        # Annualization factor (assume daily data)
        ann_factor = 252

        total_return = (equity_curve.iloc[-1] / self.config.initial_capital) - 1
        ann_return = (1 + total_return) ** (ann_factor / len(returns)) - 1
        ann_volatility = returns.std() * np.sqrt(ann_factor)

        # Sharpe ratio
        risk_free = 0.02  # 2% risk-free rate
        sharpe = (ann_return - risk_free) / ann_volatility if ann_volatility > 0 else 0

        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(ann_factor)
        sortino = (ann_return - risk_free) / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        rolling_max = equity_curve.cummax()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        if len(trades) > 1:
            # Simplified: compare entry to exit
            trade_returns = trades['price'].pct_change().dropna()
            win_rate = (trade_returns > 0).mean()
        else:
            win_rate = 0

        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'annualized_volatility': ann_volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'trades_per_year': len(trades) * ann_factor / len(returns)
        }
```

### 2. Walk-Forward Analysis

```python
from typing import List, Tuple, Callable
import pandas as pd
import numpy as np

@dataclass
class WalkForwardConfig:
    """Walk-forward analysis configuration"""
    train_periods: int = 252  # 1 year training
    test_periods: int = 63    # 3 months testing
    step_size: int = 21       # Monthly steps
    min_train_periods: int = 126  # Minimum training data
    anchored: bool = False    # Expanding vs rolling window

@dataclass
class WalkForwardResult:
    """Walk-forward analysis results"""
    in_sample_results: List[BacktestResult]
    out_of_sample_results: List[BacktestResult]
    combined_oos_equity: pd.Series
    fold_metrics: pd.DataFrame
    overfitting_ratio: float

class WalkForwardAnalysis:
    """
    Walk-forward analysis for robust strategy validation

    Prevents overfitting by testing on truly out-of-sample data
    """

    def __init__(
        self,
        config: WalkForwardConfig = None,
        backtest_engine: VectorizedBacktest = None
    ):
        self.config = config or WalkForwardConfig()
        self.engine = backtest_engine or VectorizedBacktest()

    def run(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable[[pd.DataFrame], pd.Series],
        parameter_optimizer: Optional[Callable] = None
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis

        Args:
            prices: OHLCV data
            signal_generator: Function that generates signals from prices
            parameter_optimizer: Optional function to optimize parameters on training data

        Returns:
            WalkForwardResult with in-sample and out-of-sample performance
        """
        folds = self._generate_folds(len(prices))
        in_sample_results = []
        out_of_sample_results = []

        for train_idx, test_idx in folds:
            train_data = prices.iloc[train_idx]
            test_data = prices.iloc[test_idx]

            # Optimize on training data (if provided)
            if parameter_optimizer:
                best_params = parameter_optimizer(train_data)
                train_signals = signal_generator(train_data, **best_params)
                test_signals = signal_generator(test_data, **best_params)
            else:
                train_signals = signal_generator(train_data)
                test_signals = signal_generator(test_data)

            # Run backtests
            is_result = self.engine.run(train_data, train_signals)
            oos_result = self.engine.run(test_data, test_signals)

            in_sample_results.append(is_result)
            out_of_sample_results.append(oos_result)

        # Combine out-of-sample results
        combined_oos = self._combine_oos_results(out_of_sample_results)

        # Calculate fold metrics
        fold_metrics = self._calculate_fold_metrics(
            in_sample_results, out_of_sample_results
        )

        # Calculate overfitting ratio
        overfitting_ratio = self._calculate_overfitting_ratio(
            in_sample_results, out_of_sample_results
        )

        return WalkForwardResult(
            in_sample_results=in_sample_results,
            out_of_sample_results=out_of_sample_results,
            combined_oos_equity=combined_oos,
            fold_metrics=fold_metrics,
            overfitting_ratio=overfitting_ratio
        )

    def _generate_folds(self, n_samples: int) -> List[Tuple[range, range]]:
        """Generate train/test fold indices"""
        folds = []

        if self.config.anchored:
            # Anchored: training window expands
            train_start = 0
        else:
            train_start = None  # Will be set per fold

        test_start = self.config.train_periods

        while test_start + self.config.test_periods <= n_samples:
            if not self.config.anchored:
                train_start = max(0, test_start - self.config.train_periods)

            train_end = test_start
            test_end = test_start + self.config.test_periods

            if train_end - train_start >= self.config.min_train_periods:
                folds.append((
                    range(train_start, train_end),
                    range(test_start, test_end)
                ))

            test_start += self.config.step_size

        return folds

    def _combine_oos_results(
        self,
        results: List[BacktestResult]
    ) -> pd.Series:
        """Combine out-of-sample equity curves"""
        curves = []
        for result in results:
            # Rebase each curve to start at 1
            normalized = result.equity_curve / result.equity_curve.iloc[0]
            curves.append(normalized)

        # Chain curves together
        combined = pd.concat(curves)
        combined = combined / combined.iloc[0] * self.engine.config.initial_capital

        return combined

    def _calculate_fold_metrics(
        self,
        is_results: List[BacktestResult],
        oos_results: List[BacktestResult]
    ) -> pd.DataFrame:
        """Calculate metrics for each fold"""
        rows = []

        for i, (is_res, oos_res) in enumerate(zip(is_results, oos_results)):
            rows.append({
                'fold': i,
                'is_sharpe': is_res.metrics['sharpe_ratio'],
                'oos_sharpe': oos_res.metrics['sharpe_ratio'],
                'is_return': is_res.metrics['total_return'],
                'oos_return': oos_res.metrics['total_return'],
                'is_max_dd': is_res.metrics['max_drawdown'],
                'oos_max_dd': oos_res.metrics['max_drawdown'],
                'sharpe_decay': (is_res.metrics['sharpe_ratio'] -
                                oos_res.metrics['sharpe_ratio'])
            })

        return pd.DataFrame(rows)

    def _calculate_overfitting_ratio(
        self,
        is_results: List[BacktestResult],
        oos_results: List[BacktestResult]
    ) -> float:
        """
        Calculate overfitting ratio

        Ratio > 1: Out-of-sample worse than in-sample (typical)
        Ratio > 2: Significant overfitting concern
        Ratio < 1: OOS better than IS (rare, possible luck)
        """
        is_sharpes = [r.metrics['sharpe_ratio'] for r in is_results]
        oos_sharpes = [r.metrics['sharpe_ratio'] for r in oos_results]

        avg_is_sharpe = np.mean(is_sharpes)
        avg_oos_sharpe = np.mean(oos_sharpes)

        if avg_oos_sharpe <= 0:
            return float('inf')  # Strategy doesn't work OOS

        return avg_is_sharpe / avg_oos_sharpe
```

### 3. Overfitting Detection

```python
from scipy import stats
import numpy as np

def probability_of_backtest_overfitting(
    in_sample_sharpe: float,
    out_of_sample_sharpe: float,
    n_trials: int,
    is_periods: int,
    oos_periods: int
) -> float:
    """
    Estimate probability that backtest is overfit

    Based on Bailey et al. "Probability of Backtest Overfitting"

    Args:
        in_sample_sharpe: In-sample Sharpe ratio
        out_of_sample_sharpe: Out-of-sample Sharpe ratio
        n_trials: Number of strategy variations tested
        is_periods: Number of in-sample periods
        oos_periods: Number of out-of-sample periods

    Returns:
        Probability of overfitting (0-1)
    """
    # Deflated Sharpe Ratio adjustment
    # Accounts for multiple testing
    var_sharpe = (1 + 0.5 * in_sample_sharpe ** 2) / is_periods

    # Expected maximum Sharpe from random trials
    expected_max = stats.norm.ppf(1 - 1/n_trials) * np.sqrt(var_sharpe)

    # Probability IS Sharpe is due to overfitting
    deflated_sharpe = in_sample_sharpe - expected_max

    # Compare IS to OOS
    sharpe_decay = in_sample_sharpe - out_of_sample_sharpe

    # Higher decay = more likely overfit
    pbo = stats.norm.cdf(sharpe_decay / np.sqrt(var_sharpe * 2))

    return pbo


def minimum_backtest_length(
    target_sharpe: float,
    skewness: float = 0,
    kurtosis: float = 3,
    confidence: float = 0.95
) -> int:
    """
    Calculate minimum backtest length for statistical significance

    Based on Bailey & Lopez de Prado (2012)

    Args:
        target_sharpe: Expected annualized Sharpe ratio
        skewness: Return distribution skewness
        kurtosis: Return distribution kurtosis
        confidence: Desired confidence level

    Returns:
        Minimum number of years needed
    """
    z_score = stats.norm.ppf(confidence)

    # Adjusted variance for non-normality
    variance_adjustment = 1 - skewness * target_sharpe + \
                         (kurtosis - 1) / 4 * target_sharpe ** 2

    min_years = variance_adjustment * (z_score / target_sharpe) ** 2

    return int(np.ceil(min_years))


def combinatorial_purged_cross_validation(
    data: pd.DataFrame,
    n_splits: int = 10,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Combinatorial Purged Cross-Validation (CPCV)

    More robust than standard k-fold for financial time series
    - Purging: Remove training samples close to test period
    - Embargo: Add gap between train and test
    - Combinatorial: Test all combinations of test folds

    Args:
        data: DataFrame with datetime index
        n_splits: Number of splits
        n_test_splits: Number of test splits per combination
        embargo_pct: Percentage of data to embargo after test

    Returns:
        List of (train_indices, test_indices) tuples
    """
    from itertools import combinations

    n_samples = len(data)
    indices = np.arange(n_samples)

    # Create fold boundaries
    fold_size = n_samples // n_splits
    fold_bounds = [(i * fold_size, (i + 1) * fold_size) for i in range(n_splits)]
    fold_bounds[-1] = (fold_bounds[-1][0], n_samples)  # Last fold gets remainder

    # Generate all combinations of test folds
    test_combinations = list(combinations(range(n_splits), n_test_splits))

    splits = []

    for test_folds in test_combinations:
        test_indices = []
        train_indices = []

        for fold_idx in range(n_splits):
            start, end = fold_bounds[fold_idx]
            fold_indices = indices[start:end]

            if fold_idx in test_folds:
                test_indices.extend(fold_indices)
            else:
                train_indices.extend(fold_indices)

        # Apply embargo
        embargo_size = int(n_samples * embargo_pct)
        test_start = min(test_indices)
        test_end = max(test_indices)

        # Remove training samples near test period
        train_indices = [i for i in train_indices
                        if i < test_start - embargo_size or i > test_end + embargo_size]

        splits.append((np.array(train_indices), np.array(test_indices)))

    return splits
```

### 4. Transaction Cost Modeling

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class TransactionCostModel:
    """
    Realistic transaction cost model

    Components:
    - Commission (broker fee)
    - Spread cost (bid-ask)
    - Market impact (price movement from order)
    - Slippage (execution vs. expected)
    """
    commission_per_share: float = 0.005
    commission_min: float = 1.0
    commission_max: float = 50.0

    # Spread model (relative to price)
    base_spread_bps: float = 5.0  # 5 bps base spread
    spread_vol_multiplier: float = 2.0  # Spread widens with volatility

    # Market impact model (square root)
    impact_coefficient: float = 0.1
    daily_volume: int = 1_000_000  # For impact calculation

    # Slippage
    base_slippage_bps: float = 2.0
    slippage_vol_multiplier: float = 1.0

    def calculate_total_cost(
        self,
        quantity: int,
        price: float,
        volatility: float = 0.02,
        urgency: float = 0.5
    ) -> dict:
        """
        Calculate total transaction cost

        Args:
            quantity: Order quantity (shares)
            price: Current price
            volatility: Daily volatility (decimal)
            urgency: Execution urgency 0-1

        Returns:
            Cost breakdown dictionary
        """
        trade_value = quantity * price

        # Commission
        commission = np.clip(
            quantity * self.commission_per_share,
            self.commission_min,
            self.commission_max
        )

        # Spread cost (half spread for one-way trade)
        spread_bps = self.base_spread_bps * (1 + self.spread_vol_multiplier * volatility / 0.02)
        spread_cost = trade_value * spread_bps / 10000 / 2

        # Market impact (square root model)
        participation_rate = quantity / self.daily_volume
        impact_bps = self.impact_coefficient * np.sqrt(participation_rate) * 10000
        impact_cost = trade_value * impact_bps / 10000

        # Slippage (increases with urgency and volatility)
        slippage_bps = self.base_slippage_bps * (1 + urgency) * \
                       (1 + self.slippage_vol_multiplier * volatility / 0.02)
        slippage_cost = trade_value * slippage_bps / 10000

        total_cost = commission + spread_cost + impact_cost + slippage_cost
        total_bps = total_cost / trade_value * 10000

        return {
            'commission': commission,
            'spread_cost': spread_cost,
            'impact_cost': impact_cost,
            'slippage_cost': slippage_cost,
            'total_cost': total_cost,
            'total_bps': total_bps,
            'cost_pct': total_cost / trade_value
        }


def apply_realistic_costs(
    signals: pd.Series,
    prices: pd.DataFrame,
    cost_model: TransactionCostModel,
    position_value: float = 10000
) -> pd.Series:
    """
    Apply realistic transaction costs to strategy returns

    Args:
        signals: Trading signals
        prices: OHLCV data with 'close' and optionally 'volume'
        cost_model: Transaction cost model
        position_value: Dollar value per position

    Returns:
        Returns series with costs applied
    """
    # Calculate raw returns
    raw_returns = prices['close'].pct_change()
    strategy_returns = signals.shift(1) * raw_returns

    # Calculate costs on trades
    position_changes = signals.diff().fillna(signals)
    trade_mask = position_changes != 0

    cost_deductions = pd.Series(0.0, index=signals.index)

    for idx in signals.index[trade_mask]:
        price = prices.loc[idx, 'close']
        quantity = int(position_value / price)

        costs = cost_model.calculate_total_cost(
            quantity=abs(quantity),
            price=price
        )

        # Deduct costs as percentage of position
        cost_deductions.loc[idx] = costs['cost_pct']

    net_returns = strategy_returns - cost_deductions

    return net_returns
```

### 5. Monte Carlo Validation

```python
def monte_carlo_permutation_test(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    n_simulations: int = 10000,
    metric: str = 'sharpe'
) -> dict:
    """
    Monte Carlo permutation test for strategy significance

    Shuffles returns to create null distribution
    Tests if strategy performance is statistically significant

    Args:
        strategy_returns: Strategy returns series
        benchmark_returns: Benchmark returns series
        n_simulations: Number of Monte Carlo simulations
        metric: Performance metric to test ('sharpe', 'return', 'sortino')

    Returns:
        p-value and null distribution statistics
    """
    def calculate_metric(returns: np.ndarray) -> float:
        if metric == 'sharpe':
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        elif metric == 'return':
            return np.mean(returns) * 252
        elif metric == 'sortino':
            downside = returns[returns < 0]
            if len(downside) == 0:
                return np.inf
            return np.mean(returns) / np.std(downside) * np.sqrt(252)

    # Actual strategy performance
    actual_metric = calculate_metric(strategy_returns.values)

    # Generate null distribution
    combined = np.concatenate([strategy_returns.values, benchmark_returns.values])
    null_metrics = []

    for _ in range(n_simulations):
        np.random.shuffle(combined)
        n = len(strategy_returns)
        shuffled_strategy = combined[:n]
        null_metrics.append(calculate_metric(shuffled_strategy))

    null_metrics = np.array(null_metrics)

    # Calculate p-value (one-tailed: strategy > null)
    p_value = np.mean(null_metrics >= actual_metric)

    return {
        'actual_metric': actual_metric,
        'p_value': p_value,
        'null_mean': np.mean(null_metrics),
        'null_std': np.std(null_metrics),
        'null_percentile_95': np.percentile(null_metrics, 95),
        'null_percentile_99': np.percentile(null_metrics, 99),
        'significant_95': p_value < 0.05,
        'significant_99': p_value < 0.01
    }


def bootstrap_confidence_interval(
    returns: pd.Series,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    metric_func: Callable = None
) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for strategy metrics

    Args:
        returns: Strategy returns
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        metric_func: Function to calculate metric (default: Sharpe)

    Returns:
        (lower_bound, upper_bound)
    """
    if metric_func is None:
        metric_func = lambda r: np.mean(r) / np.std(r) * np.sqrt(252)

    n = len(returns)
    bootstrap_metrics = []

    for _ in range(n_bootstrap):
        # Sample with replacement
        sample_idx = np.random.choice(n, size=n, replace=True)
        sample_returns = returns.iloc[sample_idx].values
        bootstrap_metrics.append(metric_func(sample_returns))

    alpha = 1 - confidence
    lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)

    return lower, upper
```

## Production Backtest Framework

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np

class ProductionBacktester:
    """
    Production-grade backtesting framework

    Features:
    - Walk-forward validation
    - Realistic transaction costs
    - Overfitting detection
    - Statistical significance testing
    """

    def __init__(
        self,
        backtest_config: BacktestConfig = None,
        walkforward_config: WalkForwardConfig = None,
        cost_model: TransactionCostModel = None
    ):
        self.bt_config = backtest_config or BacktestConfig()
        self.wf_config = walkforward_config or WalkForwardConfig()
        self.cost_model = cost_model or TransactionCostModel()

        self.engine = VectorizedBacktest(self.bt_config)
        self.walkforward = WalkForwardAnalysis(self.wf_config, self.engine)

    def full_validation(
        self,
        prices: pd.DataFrame,
        signal_generator: Callable,
        benchmark_returns: Optional[pd.Series] = None,
        n_monte_carlo: int = 1000
    ) -> Dict:
        """
        Run full validation suite

        Returns comprehensive validation report
        """
        # 1. Walk-forward analysis
        wf_result = self.walkforward.run(prices, signal_generator)

        # 2. Overfitting probability
        avg_is_sharpe = np.mean([r.metrics['sharpe_ratio']
                                for r in wf_result.in_sample_results])
        avg_oos_sharpe = np.mean([r.metrics['sharpe_ratio']
                                 for r in wf_result.out_of_sample_results])

        pbo = probability_of_backtest_overfitting(
            in_sample_sharpe=avg_is_sharpe,
            out_of_sample_sharpe=avg_oos_sharpe,
            n_trials=1,  # Single strategy
            is_periods=self.wf_config.train_periods,
            oos_periods=self.wf_config.test_periods
        )

        # 3. Monte Carlo significance test
        combined_oos_returns = pd.concat([
            r.returns for r in wf_result.out_of_sample_results
        ])

        if benchmark_returns is None:
            benchmark_returns = prices['close'].pct_change().dropna()

        mc_result = monte_carlo_permutation_test(
            combined_oos_returns,
            benchmark_returns,
            n_simulations=n_monte_carlo
        )

        # 4. Bootstrap confidence interval
        ci_lower, ci_upper = bootstrap_confidence_interval(combined_oos_returns)

        # 5. Minimum backtest length check
        min_years = minimum_backtest_length(target_sharpe=avg_oos_sharpe)
        actual_years = len(prices) / 252
        sufficient_data = actual_years >= min_years

        return {
            'walk_forward': {
                'n_folds': len(wf_result.in_sample_results),
                'avg_is_sharpe': avg_is_sharpe,
                'avg_oos_sharpe': avg_oos_sharpe,
                'overfitting_ratio': wf_result.overfitting_ratio,
                'fold_metrics': wf_result.fold_metrics.to_dict()
            },
            'overfitting': {
                'probability': pbo,
                'concern_level': 'high' if pbo > 0.5 else 'medium' if pbo > 0.3 else 'low'
            },
            'statistical_significance': {
                'p_value': mc_result['p_value'],
                'significant_95': mc_result['significant_95'],
                'significant_99': mc_result['significant_99'],
                'null_sharpe_95pct': mc_result['null_percentile_95']
            },
            'confidence_interval': {
                'sharpe_lower': ci_lower,
                'sharpe_upper': ci_upper,
                'confidence': 0.95
            },
            'data_sufficiency': {
                'actual_years': actual_years,
                'minimum_years': min_years,
                'sufficient': sufficient_data
            },
            'recommendation': self._generate_recommendation(
                pbo, mc_result, avg_oos_sharpe, sufficient_data
            )
        }

    def _generate_recommendation(
        self,
        pbo: float,
        mc_result: dict,
        oos_sharpe: float,
        sufficient_data: bool
    ) -> str:
        """Generate human-readable recommendation"""
        issues = []

        if pbo > 0.5:
            issues.append("High probability of overfitting")
        if not mc_result['significant_95']:
            issues.append("Not statistically significant at 95%")
        if oos_sharpe < 0.5:
            issues.append("Out-of-sample Sharpe below 0.5")
        if not sufficient_data:
            issues.append("Insufficient backtest data")

        if not issues:
            return "PASS: Strategy shows robust out-of-sample performance"
        elif len(issues) == 1:
            return f"CAUTION: {issues[0]}"
        else:
            return f"FAIL: Multiple concerns - {'; '.join(issues)}"


# Usage Example
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_days = 252 * 5  # 5 years

    prices = pd.DataFrame({
        'open': np.random.randn(n_days).cumsum() + 100,
        'high': lambda df: df['open'] + np.random.rand(n_days),
        'low': lambda df: df['open'] - np.random.rand(n_days),
        'close': np.random.randn(n_days).cumsum() + 100,
        'volume': np.random.randint(100000, 1000000, n_days)
    }, index=pd.date_range('2019-01-01', periods=n_days, freq='D'))

    # Simple momentum signal generator
    def momentum_signal(data: pd.DataFrame, lookback: int = 20) -> pd.Series:
        returns = data['close'].pct_change(lookback)
        signal = np.sign(returns)
        return signal

    # Run full validation
    backtester = ProductionBacktester()
    report = backtester.full_validation(
        prices=prices,
        signal_generator=momentum_signal,
        n_monte_carlo=1000
    )

    print("Validation Report:")
    print(f"  OOS Sharpe: {report['walk_forward']['avg_oos_sharpe']:.2f}")
    print(f"  Overfitting Prob: {report['overfitting']['probability']:.1%}")
    print(f"  P-Value: {report['statistical_significance']['p_value']:.3f}")
    print(f"  Recommendation: {report['recommendation']}")
```

## Best Practices

1. **Always use walk-forward analysis** - never trust in-sample results alone
2. **Model realistic transaction costs** - they can eliminate strategy profitability
3. **Test statistical significance** - Monte Carlo and bootstrap methods
4. **Check data sufficiency** - ensure enough history for target Sharpe
5. **Monitor overfitting ratio** - IS/OOS Sharpe ratio > 2 is concerning
6. **Use combinatorial purged CV** for parameter optimization

## Common Pitfalls

❌ **Optimizing on full dataset** then testing on same data
✅ Strict train/test separation with walk-forward

❌ **Ignoring transaction costs** in strategy evaluation
✅ Model all cost components (spread, impact, slippage)

❌ **Testing many strategies** and reporting the best
✅ Apply multiple testing corrections (Bonferroni, FDR)

❌ **Using insufficient data** for statistical significance
✅ Calculate minimum backtest length for target Sharpe

❌ **Point estimate only** for Sharpe ratio
✅ Report confidence intervals from bootstrap

## Quality Standards

- **OOS Sharpe**: Must be >0.5 for production consideration
- **Overfitting Ratio**: Must be <2.0 (IS Sharpe / OOS Sharpe)
- **Statistical Significance**: p < 0.05 on permutation test
- **Data Length**: Minimum years for target Sharpe satisfied
- **Cost Modeling**: All cost components explicitly modeled

---

**Skill Type**: Finance - Strategy Validation
**Complexity**: Complex
**Typical Usage**: Activated when trading-strategy-architect needs robust backtesting
**Performance**: Vectorized engine processes >100,000 bars/second
