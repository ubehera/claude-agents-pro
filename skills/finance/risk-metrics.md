---
name: risk-metrics
description: Load when user needs VaR, CVaR, drawdown analysis, Sharpe ratio, Sortino ratio, Calmar ratio, or portfolio risk measurement
trigger_keywords: [var, value at risk, cvar, expected shortfall, drawdown, max drawdown, sharpe ratio, sortino ratio, calmar ratio, information ratio, risk metrics, risk measurement, tail risk, downside risk, volatility, beta, alpha, treynor ratio]
---

# Risk Metrics Skill

Production-grade risk metrics including Value-at-Risk (VaR), Conditional VaR, drawdown analysis, and comprehensive risk-adjusted performance measurement.

## Core Concepts

### Risk Metric Categories

```yaml
Volatility Metrics:
  - Standard Deviation (symmetric)
  - Downside Deviation (asymmetric)
  - Semi-Variance (below-mean only)
  - Historical Volatility (realized)
  - Implied Volatility (forward-looking)

Tail Risk Metrics:
  - Value-at-Risk (VaR)
  - Conditional VaR / Expected Shortfall (CVaR)
  - Maximum Drawdown
  - Tail Ratio

Risk-Adjusted Returns:
  - Sharpe Ratio
  - Sortino Ratio
  - Calmar Ratio
  - Information Ratio
  - Treynor Ratio

Relative Risk:
  - Beta
  - Tracking Error
  - Active Share
  - Correlation
```

### Risk Metric Interpretation

```yaml
Sharpe Ratio:
  <0: Negative risk-adjusted return
  0-0.5: Sub-par
  0.5-1.0: Acceptable
  1.0-2.0: Good
  >2.0: Excellent (verify not overfit)

Maximum Drawdown:
  <10%: Low risk
  10-20%: Moderate risk
  20-30%: High risk
  >30%: Very high risk

VaR (95%, 1-day):
  Typical: 1-3% for diversified portfolio
  High: >5% indicates concentrated risk
```

## Implementation Patterns

### 1. Volatility Metrics

```python
import numpy as np
import pandas as pd
from typing import Optional, Literal
from scipy import stats

def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Annualized volatility (standard deviation)

    Args:
        returns: Return series (daily, weekly, etc.)
        periods_per_year: Number of periods per year

    Returns:
        Annualized volatility as decimal
    """
    return returns.std() * np.sqrt(periods_per_year)


def downside_deviation(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Downside deviation (semi-deviation below target)

    Only considers returns below target (usually 0 or MAR)
    Better measure for asymmetric return distributions
    """
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return 0.0

    # Squared deviations from target
    squared_deviations = (downside_returns - target_return) ** 2
    downside_var = squared_deviations.mean()

    return np.sqrt(downside_var) * np.sqrt(periods_per_year)


def exponential_volatility(
    returns: pd.Series,
    halflife: int = 20,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Exponentially-weighted volatility

    More responsive to recent volatility changes
    """
    ewm_var = returns.ewm(halflife=halflife).var()
    ewm_vol = np.sqrt(ewm_var) * np.sqrt(periods_per_year)
    return ewm_vol


def realized_volatility(
    prices: pd.Series,
    window: int = 20,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Rolling realized volatility

    Standard rolling window volatility calculation
    """
    returns = prices.pct_change()
    rolling_std = returns.rolling(window=window).std()
    return rolling_std * np.sqrt(periods_per_year)


def parkinson_volatility(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Parkinson volatility estimator (uses high-low range)

    More efficient than close-to-close volatility
    Assumes continuous trading, no gaps
    """
    log_hl = np.log(high / low)
    factor = 1 / (4 * np.log(2))

    rolling_var = (log_hl ** 2).rolling(window=window).mean() * factor
    return np.sqrt(rolling_var * periods_per_year)


def garman_klass_volatility(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
    periods_per_year: int = 252
) -> pd.Series:
    """
    Garman-Klass volatility estimator (uses OHLC)

    Most efficient estimator using OHLC data
    5-8x more efficient than close-to-close
    """
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_) ** 2

    # Garman-Klass formula
    gk_var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co

    rolling_var = gk_var.rolling(window=window).mean()
    return np.sqrt(rolling_var * periods_per_year)
```

### 2. Value-at-Risk (VaR)

```python
from typing import Literal
from scipy import stats
import numpy as np

def var_historical(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Historical VaR (non-parametric)

    Simply the percentile of historical returns
    No distributional assumptions
    """
    return -np.percentile(returns, (1 - confidence) * 100)


def var_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    distribution: Literal['normal', 't'] = 'normal'
) -> float:
    """
    Parametric VaR (assumes distribution)

    Args:
        returns: Return series
        confidence: Confidence level (0.95 = 95%)
        distribution: 'normal' or 't' (Student's t)

    Returns:
        VaR as positive number (loss)
    """
    mean = returns.mean()
    std = returns.std()

    if distribution == 'normal':
        z_score = stats.norm.ppf(1 - confidence)
    else:  # Student's t
        # Fit t-distribution
        params = stats.t.fit(returns)
        z_score = stats.t.ppf(1 - confidence, *params)

    var = -(mean + z_score * std)
    return max(0, var)


def var_monte_carlo(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10000,
    horizon_days: int = 1
) -> float:
    """
    Monte Carlo VaR

    Simulates future returns using historical distribution
    Can model non-linear portfolios
    """
    mean = returns.mean()
    std = returns.std()

    # Generate simulated returns
    simulated = np.random.normal(
        mean * horizon_days,
        std * np.sqrt(horizon_days),
        n_simulations
    )

    return -np.percentile(simulated, (1 - confidence) * 100)


def var_cornish_fisher(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Cornish-Fisher VaR (adjusts for skewness and kurtosis)

    More accurate for non-normal distributions
    """
    mean = returns.mean()
    std = returns.std()
    skew = stats.skew(returns)
    kurt = stats.kurtosis(returns)  # Excess kurtosis

    # Standard normal quantile
    z = stats.norm.ppf(1 - confidence)

    # Cornish-Fisher expansion
    z_cf = (z +
            (z**2 - 1) * skew / 6 +
            (z**3 - 3*z) * kurt / 24 -
            (2*z**3 - 5*z) * skew**2 / 36)

    var = -(mean + z_cf * std)
    return max(0, var)


class VaRCalculator:
    """
    Comprehensive VaR calculator with multiple methods
    """

    def __init__(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
        periods_per_year: int = 252
    ):
        self.returns = returns
        self.confidence = confidence
        self.periods_per_year = periods_per_year

    def calculate_all(self) -> dict:
        """Calculate VaR using all methods"""
        return {
            'historical': var_historical(self.returns, self.confidence),
            'parametric_normal': var_parametric(self.returns, self.confidence, 'normal'),
            'parametric_t': var_parametric(self.returns, self.confidence, 't'),
            'cornish_fisher': var_cornish_fisher(self.returns, self.confidence),
            'monte_carlo': var_monte_carlo(self.returns, self.confidence)
        }

    def rolling_var(
        self,
        window: int = 252,
        method: str = 'historical'
    ) -> pd.Series:
        """Calculate rolling VaR"""
        var_func = {
            'historical': var_historical,
            'parametric': lambda r, c: var_parametric(r, c, 'normal'),
            'cornish_fisher': var_cornish_fisher
        }[method]

        rolling_var = self.returns.rolling(window=window).apply(
            lambda x: var_func(x, self.confidence)
        )
        return rolling_var

    def var_breach_analysis(self, var_series: pd.Series) -> dict:
        """
        Analyze VaR breaches (actual loss > VaR)

        Good VaR model: breach rate ≈ (1 - confidence)
        """
        actual_losses = -self.returns
        breaches = actual_losses > var_series

        expected_breach_rate = 1 - self.confidence
        actual_breach_rate = breaches.mean()

        return {
            'expected_breach_rate': expected_breach_rate,
            'actual_breach_rate': actual_breach_rate,
            'n_breaches': breaches.sum(),
            'breach_ratio': actual_breach_rate / expected_breach_rate,
            'model_quality': (
                'good' if 0.8 < actual_breach_rate / expected_breach_rate < 1.2
                else 'poor'
            )
        }
```

### 3. Conditional VaR (Expected Shortfall)

```python
def cvar_historical(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Historical CVaR (Expected Shortfall)

    Average loss in the worst (1-confidence)% of cases
    More coherent risk measure than VaR
    """
    var = var_historical(returns, confidence)
    tail_losses = -returns[returns <= -var]

    if len(tail_losses) == 0:
        return var

    return tail_losses.mean()


def cvar_parametric(
    returns: pd.Series,
    confidence: float = 0.95,
    distribution: Literal['normal', 't'] = 'normal'
) -> float:
    """
    Parametric CVaR (Expected Shortfall)

    For normal distribution:
    CVaR = μ + σ × φ(z) / (1 - α)

    Where φ is PDF and z is the VaR quantile
    """
    mean = returns.mean()
    std = returns.std()

    if distribution == 'normal':
        z = stats.norm.ppf(1 - confidence)
        phi_z = stats.norm.pdf(z)
        cvar = -(mean + std * phi_z / (1 - confidence))
    else:  # Student's t
        params = stats.t.fit(returns)
        df = params[0]
        z = stats.t.ppf(1 - confidence, df)
        # t-distribution CVaR formula
        cvar = -(mean + std * (
            stats.t.pdf(z, df) * (df + z**2) / ((df - 1) * (1 - confidence))
        ))

    return max(0, cvar)


def cvar_monte_carlo(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10000
) -> float:
    """
    Monte Carlo CVaR

    Average of simulated losses beyond VaR
    """
    mean = returns.mean()
    std = returns.std()

    simulated = np.random.normal(mean, std, n_simulations)
    var = -np.percentile(simulated, (1 - confidence) * 100)

    tail_losses = -simulated[simulated <= -var]
    return tail_losses.mean()
```

### 4. Drawdown Analysis

```python
from dataclasses import dataclass
from typing import List, Tuple
import pandas as pd
import numpy as np

@dataclass
class DrawdownInfo:
    """Information about a single drawdown"""
    start_date: pd.Timestamp
    trough_date: pd.Timestamp
    end_date: Optional[pd.Timestamp]
    peak_value: float
    trough_value: float
    drawdown_pct: float
    duration_days: int
    recovery_days: Optional[int]

def calculate_drawdowns(equity_curve: pd.Series) -> pd.Series:
    """
    Calculate drawdown series

    Returns drawdown at each point as negative percentage
    """
    rolling_max = equity_curve.cummax()
    drawdowns = (equity_curve - rolling_max) / rolling_max
    return drawdowns


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as positive percentage"""
    drawdowns = calculate_drawdowns(equity_curve)
    return abs(drawdowns.min())


def average_drawdown(equity_curve: pd.Series) -> float:
    """Average drawdown (only drawdown periods)"""
    drawdowns = calculate_drawdowns(equity_curve)
    dd_periods = drawdowns[drawdowns < 0]
    return abs(dd_periods.mean()) if len(dd_periods) > 0 else 0.0


def drawdown_duration(equity_curve: pd.Series) -> Tuple[int, int]:
    """
    Calculate max and average drawdown duration

    Returns:
        (max_duration_days, avg_duration_days)
    """
    drawdowns = calculate_drawdowns(equity_curve)

    # Find drawdown periods
    in_drawdown = drawdowns < 0
    drawdown_changes = in_drawdown.astype(int).diff()

    starts = drawdowns.index[drawdown_changes == 1]
    ends = drawdowns.index[drawdown_changes == -1]

    if len(starts) == 0:
        return 0, 0

    # Handle ongoing drawdown
    if len(ends) < len(starts):
        ends = ends.append(pd.Index([drawdowns.index[-1]]))

    durations = [(e - s).days for s, e in zip(starts, ends)]

    return max(durations), int(np.mean(durations))


def analyze_drawdowns(
    equity_curve: pd.Series,
    top_n: int = 5
) -> List[DrawdownInfo]:
    """
    Detailed analysis of top N drawdowns

    Returns list of DrawdownInfo objects
    """
    drawdowns = calculate_drawdowns(equity_curve)
    rolling_max = equity_curve.cummax()

    # Find drawdown troughs
    in_drawdown = drawdowns < 0
    drawdown_changes = in_drawdown.astype(int).diff()

    starts = drawdowns.index[drawdown_changes == 1].tolist()
    ends = drawdowns.index[drawdown_changes == -1].tolist()

    if len(starts) == 0:
        return []

    # Handle ongoing drawdown
    ongoing = len(ends) < len(starts)
    if ongoing:
        ends.append(None)

    drawdown_list = []

    for start, end in zip(starts, ends):
        if end is not None:
            period = drawdowns[start:end]
        else:
            period = drawdowns[start:]

        trough_idx = period.idxmin()
        trough_dd = period[trough_idx]

        peak_value = rolling_max[start]
        trough_value = equity_curve[trough_idx]

        # Calculate recovery time
        if end is not None:
            recovery_days = (end - trough_idx).days
            duration = (end - start).days
        else:
            recovery_days = None
            duration = (equity_curve.index[-1] - start).days

        drawdown_list.append(DrawdownInfo(
            start_date=start,
            trough_date=trough_idx,
            end_date=end,
            peak_value=peak_value,
            trough_value=trough_value,
            drawdown_pct=abs(trough_dd),
            duration_days=duration,
            recovery_days=recovery_days
        ))

    # Sort by severity and return top N
    drawdown_list.sort(key=lambda x: x.drawdown_pct, reverse=True)
    return drawdown_list[:top_n]


def ulcer_index(equity_curve: pd.Series) -> float:
    """
    Ulcer Index - measure of downside volatility

    Square root of mean squared drawdown
    Lower is better
    """
    drawdowns = calculate_drawdowns(equity_curve)
    squared_dd = drawdowns ** 2
    return np.sqrt(squared_dd.mean())
```

### 5. Risk-Adjusted Return Metrics

```python
def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Sharpe Ratio

    Excess return per unit of total risk
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = excess_returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)

    if ann_vol == 0:
        return 0.0

    return ann_return / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Sortino Ratio

    Excess return per unit of downside risk
    Better than Sharpe for asymmetric returns
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = excess_returns.mean() * periods_per_year

    downside_dev = downside_deviation(returns, target_return, periods_per_year)

    if downside_dev == 0:
        return 0.0

    return ann_return / downside_dev


def calmar_ratio(
    returns: pd.Series,
    equity_curve: Optional[pd.Series] = None,
    periods_per_year: int = 252
) -> float:
    """
    Calmar Ratio

    Annualized return / Maximum drawdown
    """
    ann_return = returns.mean() * periods_per_year

    if equity_curve is None:
        equity_curve = (1 + returns).cumprod()

    mdd = max_drawdown(equity_curve)

    if mdd == 0:
        return 0.0

    return ann_return / mdd


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Information Ratio

    Active return / Tracking error
    Measures skill relative to benchmark
    """
    active_returns = returns - benchmark_returns
    ann_active_return = active_returns.mean() * periods_per_year
    tracking_error = active_returns.std() * np.sqrt(periods_per_year)

    if tracking_error == 0:
        return 0.0

    return ann_active_return / tracking_error


def treynor_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> float:
    """
    Treynor Ratio

    Excess return / Beta
    Risk-adjusted return per unit of systematic risk
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_excess = excess_returns.mean() * periods_per_year

    # Calculate beta
    covariance = np.cov(returns, benchmark_returns)[0, 1]
    benchmark_var = benchmark_returns.var()
    beta = covariance / benchmark_var if benchmark_var > 0 else 1.0

    if beta == 0:
        return 0.0

    return ann_excess / beta


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0
) -> float:
    """
    Omega Ratio

    Probability-weighted ratio of gains vs losses
    above/below threshold

    Omega > 1: More upside than downside
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]

    sum_gains = gains.sum()
    sum_losses = losses.sum()

    if sum_losses == 0:
        return float('inf') if sum_gains > 0 else 1.0

    return sum_gains / sum_losses


def tail_ratio(
    returns: pd.Series,
    percentile: float = 0.05
) -> float:
    """
    Tail Ratio

    Right tail / Left tail at given percentile
    Measures asymmetry in tail risk

    >1: Positive skew (larger gains than losses)
    <1: Negative skew (larger losses than gains)
    """
    right_tail = np.percentile(returns, 100 - percentile * 100)
    left_tail = abs(np.percentile(returns, percentile * 100))

    if left_tail == 0:
        return float('inf') if right_tail > 0 else 1.0

    return right_tail / left_tail
```

## Production Risk Dashboard

```python
from dataclasses import dataclass
from typing import Dict, Optional
import pandas as pd
import numpy as np

@dataclass
class RiskReport:
    """Comprehensive risk report"""
    volatility_metrics: Dict[str, float]
    var_metrics: Dict[str, float]
    drawdown_metrics: Dict[str, any]
    risk_adjusted_metrics: Dict[str, float]
    relative_metrics: Optional[Dict[str, float]]
    risk_budget: Dict[str, float]

class RiskAnalyzer:
    """
    Production risk analysis dashboard

    Calculates comprehensive risk metrics for portfolios
    """

    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        self.returns = returns
        self.benchmark = benchmark_returns
        self.rf = risk_free_rate
        self.periods = periods_per_year

        # Calculate equity curve
        self.equity = (1 + returns).cumprod()

    def full_report(self) -> RiskReport:
        """Generate comprehensive risk report"""
        return RiskReport(
            volatility_metrics=self._volatility_metrics(),
            var_metrics=self._var_metrics(),
            drawdown_metrics=self._drawdown_metrics(),
            risk_adjusted_metrics=self._risk_adjusted_metrics(),
            relative_metrics=self._relative_metrics() if self.benchmark is not None else None,
            risk_budget=self._risk_budget()
        )

    def _volatility_metrics(self) -> Dict[str, float]:
        """Calculate volatility metrics"""
        return {
            'annualized_volatility': annualized_volatility(self.returns, self.periods),
            'downside_deviation': downside_deviation(self.returns, 0.0, self.periods),
            'current_ewm_vol': exponential_volatility(self.returns, 20, self.periods).iloc[-1],
            'vol_30d': self.returns.tail(30).std() * np.sqrt(self.periods),
            'vol_90d': self.returns.tail(90).std() * np.sqrt(self.periods)
        }

    def _var_metrics(self) -> Dict[str, float]:
        """Calculate VaR and CVaR metrics"""
        var_calc = VaRCalculator(self.returns, 0.95, self.periods)
        all_var = var_calc.calculate_all()

        return {
            'var_95_historical': all_var['historical'],
            'var_95_parametric': all_var['parametric_normal'],
            'var_95_cornish_fisher': all_var['cornish_fisher'],
            'cvar_95_historical': cvar_historical(self.returns, 0.95),
            'var_99_historical': var_historical(self.returns, 0.99),
            'cvar_99_historical': cvar_historical(self.returns, 0.99)
        }

    def _drawdown_metrics(self) -> Dict[str, any]:
        """Calculate drawdown metrics"""
        max_dur, avg_dur = drawdown_duration(self.equity)
        top_dd = analyze_drawdowns(self.equity, top_n=3)

        return {
            'max_drawdown': max_drawdown(self.equity),
            'average_drawdown': average_drawdown(self.equity),
            'max_drawdown_duration_days': max_dur,
            'avg_drawdown_duration_days': avg_dur,
            'ulcer_index': ulcer_index(self.equity),
            'current_drawdown': calculate_drawdowns(self.equity).iloc[-1],
            'top_3_drawdowns': [
                {
                    'start': str(dd.start_date.date()),
                    'trough': str(dd.trough_date.date()),
                    'drawdown_pct': f"{dd.drawdown_pct:.2%}",
                    'duration_days': dd.duration_days
                }
                for dd in top_dd
            ]
        }

    def _risk_adjusted_metrics(self) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""
        metrics = {
            'sharpe_ratio': sharpe_ratio(self.returns, self.rf, self.periods),
            'sortino_ratio': sortino_ratio(self.returns, self.rf, 0.0, self.periods),
            'calmar_ratio': calmar_ratio(self.returns, self.equity, self.periods),
            'omega_ratio': omega_ratio(self.returns, 0.0),
            'tail_ratio': tail_ratio(self.returns, 0.05)
        }

        # Rolling Sharpe
        rolling_sharpe = self.returns.rolling(63).apply(
            lambda x: sharpe_ratio(x, self.rf, self.periods)
        )
        metrics['rolling_sharpe_3m'] = rolling_sharpe.iloc[-1]

        return metrics

    def _relative_metrics(self) -> Dict[str, float]:
        """Calculate metrics relative to benchmark"""
        if self.benchmark is None:
            return {}

        # Beta
        covariance = np.cov(self.returns, self.benchmark)[0, 1]
        benchmark_var = self.benchmark.var()
        beta = covariance / benchmark_var if benchmark_var > 0 else 1.0

        # Alpha
        portfolio_return = self.returns.mean() * self.periods
        benchmark_return = self.benchmark.mean() * self.periods
        alpha = portfolio_return - (self.rf + beta * (benchmark_return - self.rf))

        # Correlation
        correlation = self.returns.corr(self.benchmark)

        return {
            'beta': beta,
            'alpha': alpha,
            'correlation': correlation,
            'information_ratio': information_ratio(self.returns, self.benchmark, self.periods),
            'treynor_ratio': treynor_ratio(self.returns, self.benchmark, self.rf, self.periods),
            'tracking_error': (self.returns - self.benchmark).std() * np.sqrt(self.periods)
        }

    def _risk_budget(self) -> Dict[str, float]:
        """Calculate risk budget allocation"""
        total_vol = annualized_volatility(self.returns, self.periods)
        var_95 = var_historical(self.returns, 0.95)
        mdd = max_drawdown(self.equity)

        return {
            'volatility_contribution': total_vol,
            'var_95_daily': var_95,
            'var_95_weekly': var_95 * np.sqrt(5),
            'var_95_monthly': var_95 * np.sqrt(21),
            'max_drawdown_capacity': mdd,
            'kelly_fraction': self._kelly_criterion()
        }

    def _kelly_criterion(self) -> float:
        """
        Kelly Criterion for optimal position sizing

        f* = μ / σ² (simplified version)
        """
        mean_return = self.returns.mean() * self.periods
        variance = self.returns.var() * self.periods

        if variance == 0:
            return 0.0

        kelly = mean_return / variance

        # Cap at reasonable levels
        return np.clip(kelly, 0, 2.0)

    def stress_test(
        self,
        scenarios: Dict[str, float]  # {'scenario_name': return_shock}
    ) -> Dict[str, Dict[str, float]]:
        """
        Stress test portfolio under various scenarios

        Args:
            scenarios: Dict of scenario names to return shocks

        Returns:
            Impact on portfolio value and risk metrics
        """
        results = {}
        current_value = self.equity.iloc[-1]

        for name, shock in scenarios.items():
            shocked_value = current_value * (1 + shock)
            value_change = shocked_value - current_value

            results[name] = {
                'return_shock': shock,
                'value_impact': value_change,
                'value_impact_pct': shock,
                'new_drawdown': max(0, -shock + calculate_drawdowns(self.equity).iloc[-1])
            }

        return results


# Usage Example
if __name__ == "__main__":
    # Generate sample returns
    np.random.seed(42)
    n_days = 252 * 3

    # Strategy returns (slightly positive drift, realistic vol)
    returns = pd.Series(
        np.random.normal(0.0004, 0.015, n_days),  # ~10% annual return, 24% vol
        index=pd.date_range('2021-01-01', periods=n_days, freq='D')
    )

    # Benchmark returns
    benchmark = pd.Series(
        np.random.normal(0.0003, 0.012, n_days),  # ~8% annual return, 19% vol
        index=returns.index
    )

    # Create analyzer
    analyzer = RiskAnalyzer(
        returns=returns,
        benchmark_returns=benchmark,
        risk_free_rate=0.02
    )

    # Generate full report
    report = analyzer.full_report()

    print("Risk Report Summary")
    print("=" * 50)

    print("\nVolatility:")
    print(f"  Annual Vol: {report.volatility_metrics['annualized_volatility']:.1%}")
    print(f"  Downside Dev: {report.volatility_metrics['downside_deviation']:.1%}")

    print("\nValue-at-Risk (95%):")
    print(f"  Historical: {report.var_metrics['var_95_historical']:.2%}")
    print(f"  CVaR: {report.var_metrics['cvar_95_historical']:.2%}")

    print("\nDrawdown:")
    print(f"  Maximum: {report.drawdown_metrics['max_drawdown']:.1%}")
    print(f"  Current: {report.drawdown_metrics['current_drawdown']:.1%}")

    print("\nRisk-Adjusted Returns:")
    print(f"  Sharpe: {report.risk_adjusted_metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino: {report.risk_adjusted_metrics['sortino_ratio']:.2f}")
    print(f"  Calmar: {report.risk_adjusted_metrics['calmar_ratio']:.2f}")

    if report.relative_metrics:
        print("\nRelative Metrics:")
        print(f"  Beta: {report.relative_metrics['beta']:.2f}")
        print(f"  Alpha: {report.relative_metrics['alpha']:.1%}")
        print(f"  Info Ratio: {report.relative_metrics['information_ratio']:.2f}")

    # Stress test
    stress_scenarios = {
        'Market Crash (-20%)': -0.20,
        'Correction (-10%)': -0.10,
        'Flash Crash (-5%)': -0.05,
        'Rally (+10%)': 0.10
    }

    print("\nStress Test Results:")
    stress_results = analyzer.stress_test(stress_scenarios)
    for scenario, result in stress_results.items():
        print(f"  {scenario}: {result['value_impact_pct']:+.1%}")
```

## Best Practices

1. **Use multiple VaR methods** and compare results
2. **Include CVaR** - more coherent than VaR for tail risk
3. **Analyze drawdown characteristics** - not just max drawdown
4. **Consider time-varying volatility** with EWMA or GARCH
5. **Stress test regularly** with historical and hypothetical scenarios
6. **Match risk metrics to investment horizon**

## Common Pitfalls

❌ **Using VaR alone** without CVaR
✅ CVaR captures tail risk better

❌ **Assuming normal distribution** for fat-tailed returns
✅ Use Cornish-Fisher or historical VaR

❌ **Ignoring regime changes** in volatility
✅ Use rolling or exponential volatility

❌ **Single Sharpe ratio** without confidence interval
✅ Bootstrap to estimate uncertainty

❌ **Comparing Sharpe ratios** at different frequencies
✅ Annualize consistently

## Quality Standards

- **VaR Accuracy**: Breach rate within 20% of expected
- **Volatility Lag**: <5 day lag in detecting regime change
- **Drawdown Detection**: Real-time tracking with <1 day lag
- **Calculation Speed**: Full report in <100ms
- **Statistical Significance**: Bootstrap intervals for all ratios

---

**Skill Type**: Finance - Risk Measurement
**Complexity**: Complex
**Typical Usage**: Activated when trading-risk-manager needs portfolio risk analysis
**Performance**: Real-time risk calculations with sub-second latency
