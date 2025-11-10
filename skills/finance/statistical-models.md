---
name: statistical-models
description: Load when user needs statistical analysis, time-series modeling, cointegration testing, GARCH volatility models, correlation analysis, or statistical arbitrage
trigger_keywords: [statistical analysis, time series, cointegration, garch, volatility model, correlation, stationarity, adf test, kpss, arima, statistical arbitrage, pairs trading, regression, hypothesis testing]
---

# Statistical Models and Analysis Skill

Advanced statistical methods for financial time-series analysis, cointegration testing, volatility modeling, and feature engineering for quantitative trading strategies.

## Core Concepts

### Time-Series Analysis

**Stationarity Testing**:
- **ADF Test** (Augmented Dickey-Fuller): Tests null hypothesis of unit root (non-stationary)
- **KPSS Test**: Tests null hypothesis of stationarity (reverse of ADF)
- **Why it matters**: Most statistical models require stationary data

**Autocorrelation**:
- **ACF** (Autocorrelation Function): Correlation of series with its own lags
- **PACF** (Partial Autocorrelation Function): Direct correlation after removing intervening lags
- **Use cases**: Identifying AR/MA model orders, detecting mean reversion

**Models**:
- **ARIMA**: AutoRegressive Integrated Moving Average for forecasting
- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity for volatility

### Cointegration and Pairs Trading

**Concept**: Two non-stationary time series that move together (shared trend)
- Individual series: non-stationary (random walk)
- Spread between series: stationary (mean-reverting)
- **Use case**: Statistical arbitrage, pairs trading

**Tests**:
- **Engle-Granger**: Two-step cointegration test
- **Johansen**: Multivariate cointegration (>2 series)

### Correlation Analysis

**Types**:
- **Pearson**: Linear correlation (-1 to 1)
- **Spearman**: Rank correlation (non-linear relationships)
- **Rolling correlation**: Time-varying correlation windows

**Applications**:
- Portfolio diversification (find low-correlation assets)
- Pairs trading (find highly correlated pairs)
- Risk management (correlation breakdown during crises)

## Implementation Patterns

### Stationarity Testing

**1. Augmented Dickey-Fuller (ADF) Test**
```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

def test_stationarity_adf(
    series: pd.Series,
    significance_level: float = 0.05
) -> dict:
    """
    Test stationarity using ADF test

    Args:
        series: Time series data
        significance_level: Alpha for hypothesis test

    Returns:
        Dictionary with test results and interpretation
    """
    result = adfuller(series.dropna(), autolag='AIC')

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level,
        'interpretation': (
            f"{'Reject' if result[1] < significance_level else 'Fail to reject'} "
            f"null hypothesis of unit root (non-stationarity) at {significance_level} level. "
            f"Series is {'stationary' if result[1] < significance_level else 'non-stationary'}."
        )
    }

# Usage
prices = pd.Series([100, 102, 101, 105, 103, 108, 110])
adf_result = test_stationarity_adf(prices)
print(f"P-value: {adf_result['p_value']:.4f}")
print(f"Stationary: {adf_result['is_stationary']}")
```

**2. KPSS Test** (Confirmatory Test)
```python
from statsmodels.tsa.stattools import kpss

def test_stationarity_kpss(
    series: pd.Series,
    regression: str = 'c',  # 'c' for constant, 'ct' for constant + trend
    significance_level: float = 0.05
) -> dict:
    """
    Test stationarity using KPSS test
    Null hypothesis: Series is stationary

    Returns:
        Dictionary with test results
    """
    result = kpss(series.dropna(), regression=regression, nlags='auto')

    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[3],
        'is_stationary': result[1] > significance_level,  # Reversed from ADF!
        'interpretation': (
            f"{'Fail to reject' if result[1] > significance_level else 'Reject'} "
            f"null hypothesis of stationarity at {significance_level} level. "
            f"Series is {'stationary' if result[1] > significance_level else 'non-stationary'}."
        )
    }
```

### Cointegration Testing

**Engle-Granger Test for Pairs Trading**
```python
from statsmodels.tsa.stattools import coint
import numpy as np

def test_cointegration(
    series1: pd.Series,
    series2: pd.Series,
    significance_level: float = 0.05
) -> dict:
    """
    Test cointegration between two price series using Engle-Granger test

    Args:
        series1, series2: Price series (must be same length)
        significance_level: Alpha for hypothesis test

    Returns:
        Dictionary with cointegration test results and hedge ratio
    """
    # Align series
    data = pd.DataFrame({'s1': series1, 's2': series2}).dropna()

    # Cointegration test
    score, p_value, crit_values = coint(data['s1'], data['s2'])

    # Calculate hedge ratio (OLS regression slope)
    hedge_ratio = np.polyfit(data['s1'], data['s2'], 1)[0]

    # Calculate spread
    spread = data['s2'] - hedge_ratio * data['s1']

    return {
        'cointegrated': p_value < significance_level,
        'p_value': p_value,
        'test_statistic': score,
        'critical_values': crit_values,
        'hedge_ratio': hedge_ratio,
        'spread': spread,
        'spread_mean': spread.mean(),
        'spread_std': spread.std(),
        'interpretation': (
            f"Series {'are' if p_value < significance_level else 'are NOT'} "
            f"cointegrated at {significance_level} level (p={p_value:.4f}). "
            f"Hedge ratio: {hedge_ratio:.4f}"
        )
    }

# Usage Example - Pairs Trading Setup
stock_a = pd.Series([100, 102, 105, 103, 107, 110])
stock_b = pd.Series([50, 51, 52.5, 51.5, 53.5, 55])

coint_result = test_cointegration(stock_a, stock_b)
if coint_result['cointegrated']:
    print(f"✓ Cointegrated pair found (p={coint_result['p_value']:.4f})")
    print(f"Hedge ratio: {coint_result['hedge_ratio']:.4f}")
    print(f"Spread μ={coint_result['spread_mean']:.4f}, σ={coint_result['spread_std']:.4f}")
```

### GARCH Volatility Modeling

**GARCH(1,1) Implementation**
```python
from arch import arch_model
import pandas as pd

def fit_garch_model(
    returns: pd.Series,
    p: int = 1,
    q: int = 1,
    vol: str = 'GARCH'
) -> dict:
    """
    Fit GARCH model to return series

    Args:
        returns: Return series (NOT prices)
        p: GARCH lag order
        q: ARCH lag order
        vol: Volatility model ('GARCH', 'EGARCH', 'TARCH')

    Returns:
        Model fit results and forecasts
    """
    # Multiply by 100 for numerical stability
    returns_pct = returns * 100

    # Fit GARCH model
    model = arch_model(
        returns_pct,
        vol=vol,
        p=p,
        q=q,
        dist='normal'
    )

    fitted = model.fit(disp='off')

    # Forecast next period volatility
    forecast = fitted.forecast(horizon=1)
    forecasted_var = forecast.variance.iloc[-1, 0]

    # Annualize volatility (assuming daily returns)
    annual_vol = np.sqrt(forecasted_var * 252) / 100

    return {
        'model': fitted,
        'aic': fitted.aic,
        'bic': fitted.bic,
        'parameters': fitted.params.to_dict(),
        'conditional_volatility': fitted.conditional_volatility / 100,  # Back to decimal
        'forecasted_volatility_annual': annual_vol,
        'summary': fitted.summary()
    }

# Usage
returns = pd.Series(np.random.randn(252) * 0.02)  # Daily returns
garch_result = fit_garch_model(returns)
print(f"Forecasted annual volatility: {garch_result['forecasted_volatility_annual']:.2%}")
print(f"AIC: {garch_result['aic']:.2f}, BIC: {garch_result['bic']:.2f}")
```

### Correlation Analysis

**Rolling Correlation**
```python
def calculate_rolling_correlation(
    series1: pd.Series,
    series2: pd.Series,
    window: int = 60,
    method: str = 'pearson'
) -> pd.Series:
    """
    Calculate rolling correlation between two series

    Args:
        series1, series2: Time series
        window: Rolling window size (e.g., 60 days)
        method: 'pearson' or 'spearman'

    Returns:
        Rolling correlation series
    """
    data = pd.DataFrame({'s1': series1, 's2': series2})

    if method == 'pearson':
        corr = data['s1'].rolling(window).corr(data['s2'])
    elif method == 'spearman':
        corr = data['s1'].rolling(window).apply(
            lambda x: x.corr(data['s2'].iloc[x.index], method='spearman')
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr

# Usage - Detect correlation breakdown
stock1 = pd.Series(np.random.randn(252).cumsum())
stock2 = stock1 + np.random.randn(252) * 0.5  # Correlated with noise

rolling_corr = calculate_rolling_correlation(stock1, stock2, window=30)
print(f"Current 30-day correlation: {rolling_corr.iloc[-1]:.4f}")
print(f"Mean correlation: {rolling_corr.mean():.4f}")
```

**Correlation Matrix**
```python
def calculate_correlation_matrix(
    data: pd.DataFrame,
    method: str = 'pearson'
) -> pd.DataFrame:
    """
    Calculate correlation matrix for multiple series

    Args:
        data: DataFrame with multiple time series columns
        method: 'pearson' or 'spearman'

    Returns:
        Correlation matrix
    """
    return data.corr(method=method)

# Find pairs with high correlation for pairs trading
def find_cointegrated_pairs(
    prices: pd.DataFrame,
    significance_level: float = 0.05
) -> list[dict]:
    """
    Screen for cointegrated pairs in a basket of stocks

    Args:
        prices: DataFrame with stock prices (columns = tickers)
        significance_level: P-value threshold

    Returns:
        List of cointegrated pairs with statistics
    """
    n = prices.shape[1]
    tickers = prices.columns
    pairs = []

    for i in range(n):
        for j in range(i+1, n):
            result = test_cointegration(
                prices[tickers[i]],
                prices[tickers[j]],
                significance_level
            )

            if result['cointegrated']:
                pairs.append({
                    'ticker1': tickers[i],
                    'ticker2': tickers[j],
                    'p_value': result['p_value'],
                    'hedge_ratio': result['hedge_ratio'],
                    'spread_mean': result['spread_mean'],
                    'spread_std': result['spread_std']
                })

    return sorted(pairs, key=lambda x: x['p_value'])
```

## Feature Engineering

### Statistical Features for ML

```python
def engineer_statistical_features(
    prices: pd.Series,
    windows: list[int] = [5, 10, 20, 50]
) -> pd.DataFrame:
    """
    Create statistical features for machine learning trading models

    Args:
        prices: Price series
        windows: List of rolling window sizes

    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame(index=prices.index)

    # Returns
    returns = prices.pct_change()
    features['return'] = returns

    for window in windows:
        # Rolling statistics
        features[f'sma_{window}'] = prices.rolling(window).mean()
        features[f'std_{window}'] = prices.rolling(window).std()
        features[f'min_{window}'] = prices.rolling(window).min()
        features[f'max_{window}'] = prices.rolling(window).max()

        # Momentum features
        features[f'roc_{window}'] = prices.pct_change(window)
        features[f'position_{window}'] = (
            (prices - features[f'min_{window}']) /
            (features[f'max_{window}'] - features[f'min_{window}'])
        )

        # Volatility features
        features[f'realized_vol_{window}'] = (
            returns.rolling(window).std() * np.sqrt(252)
        )

        # Z-score (mean reversion signal)
        features[f'zscore_{window}'] = (
            (prices - features[f'sma_{window}']) / features[f'std_{window}']
        )

    return features.dropna()

# Usage
prices = pd.Series(np.random.randn(252).cumsum() + 100)
ml_features = engineer_statistical_features(prices)
print(f"Generated {ml_features.shape[1]} features")
print(ml_features.head())
```

### Z-Score for Mean Reversion

```python
def calculate_zscore(
    series: pd.Series,
    window: int = 20
) -> pd.Series:
    """
    Calculate rolling z-score for mean reversion signals

    Z-score interpretation:
    - >2: Significantly above mean (sell signal)
    - <-2: Significantly below mean (buy signal)
    - Between -1 and 1: Normal range

    Args:
        series: Price or spread series
        window: Rolling window for mean/std calculation

    Returns:
        Rolling z-score series
    """
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()

    zscore = (series - rolling_mean) / rolling_std
    return zscore

# Usage - Pairs Trading Signal
spread = pd.Series(np.random.randn(252).cumsum())  # Cointegrated spread
zscore = calculate_zscore(spread, window=20)

# Generate signals
signals = pd.Series(index=zscore.index, dtype=float)
signals[zscore > 2] = -1  # Short spread (overbought)
signals[zscore < -2] = 1   # Long spread (oversold)
signals[(zscore > -0.5) & (zscore < 0.5)] = 0  # Close position (mean revert)

print(f"Buy signals: {(signals == 1).sum()}")
print(f"Sell signals: {(signals == -1).sum()}")
```

## Production-Ready Class

```python
from typing import Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss, coint

class StatisticalAnalysis:
    """Statistical analysis tools for quantitative trading"""

    @staticmethod
    def test_stationarity(
        series: pd.Series,
        test: str = 'adf',
        significance: float = 0.05
    ) -> dict:
        """Test stationarity using ADF or KPSS"""
        if test == 'adf':
            result = adfuller(series.dropna(), autolag='AIC')
            return {
                'test': 'ADF',
                'statistic': result[0],
                'p_value': result[1],
                'is_stationary': result[1] < significance
            }
        elif test == 'kpss':
            result = kpss(series.dropna(), regression='c', nlags='auto')
            return {
                'test': 'KPSS',
                'statistic': result[0],
                'p_value': result[1],
                'is_stationary': result[1] > significance
            }
        else:
            raise ValueError(f"Unknown test: {test}")

    @staticmethod
    def test_cointegration(
        series1: pd.Series,
        series2: pd.Series,
        significance: float = 0.05
    ) -> dict:
        """Test cointegration for pairs trading"""
        data = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        score, p_value, _ = coint(data['s1'], data['s2'])

        hedge_ratio = np.polyfit(data['s1'], data['s2'], 1)[0]
        spread = data['s2'] - hedge_ratio * data['s1']

        return {
            'cointegrated': p_value < significance,
            'p_value': p_value,
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'spread_zscore': calculate_zscore(spread, window=20)
        }

    @staticmethod
    def rolling_correlation(
        series1: pd.Series,
        series2: pd.Series,
        window: int = 60,
        method: str = 'pearson'
    ) -> pd.Series:
        """Calculate rolling correlation"""
        data = pd.DataFrame({'s1': series1, 's2': series2})
        return data['s1'].rolling(window).corr(data['s2'])

    @staticmethod
    def calculate_zscore(
        series: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """Calculate rolling z-score"""
        mean = series.rolling(window).mean()
        std = series.rolling(window).std()
        return (series - mean) / std
```

## Best Practices

1. **Always test stationarity** before applying statistical models (ARIMA, cointegration)
2. **Use both ADF and KPSS** for robust stationarity confirmation
3. **Check cointegration p-value** before pairs trading (<0.05 threshold)
4. **Monitor rolling correlation** to detect relationship breakdown
5. **Use z-score thresholds** of ±2 for mean reversion signals
6. **Annualize volatility** consistently (daily → annual: multiply by √252)
7. **Handle missing data** with `.dropna()` before statistical tests

## Common Pitfalls

❌ **Applying ARIMA to non-stationary data**
✅ Difference the series first or test stationarity

❌ **Using correlation instead of cointegration** for pairs trading
✅ Correlation measures linear relationship, cointegration measures mean reversion

❌ **Forgetting to multiply returns by 100** in GARCH models
✅ GARCH models are more numerically stable with percentage returns

❌ **Using Pearson correlation on non-linear relationships**
✅ Use Spearman rank correlation for monotonic non-linear relationships

❌ **Not monitoring rolling statistics** (assuming static relationships)
✅ Calculate rolling correlation/cointegration to detect regime changes

## Quality Standards

- **Stationarity Tests**: ADF + KPSS both agree (avoid contradictions)
- **Cointegration**: p-value <0.05 for pairs trading, hedge ratio stable over time
- **GARCH Fit**: AIC/BIC convergence, parameters within reasonable bounds
- **Feature Engineering**: No look-ahead bias, proper handling of NaN values
- **Z-Score Signals**: Entry threshold ±2, exit threshold ±0.5, no over-trading

---

**Skill Type**: Finance - Statistical Analysis
**Complexity**: Complex
**Typical Usage**: Activated when quantitative-analyst needs time-series analysis, cointegration testing, or statistical features
**Performance**: Optimized with statsmodels and vectorized pandas operations
