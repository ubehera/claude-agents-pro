---
name: factor-investing
description: Load when user needs factor models, alpha research, Fama-French, cross-sectional analysis, or systematic factor strategies. Covers factor construction, decomposition, and timing.
trigger_keywords: [factor investing, fama french, alpha research, factor model, momentum factor, value factor, quality factor, size factor, cross-sectional, factor exposure, factor returns, smart beta, risk premia, factor timing, information ratio]
---

# Factor Investing & Alpha Research Skill

Systematic factor-based investing with multi-factor models, alpha decomposition, and factor timing strategies.

## Core Concepts

- **Risk Premium vs Alpha**: Factors represent systematic risk premia (compensation for bearing risk); true alpha is excess return unexplained by factors
- **Factor Crowding**: When too much capital chases a factor, expected returns decline and reversal risk increases
- **Factor Decay**: Factor returns weaken over time as strategies become commoditized; momentum especially prone to crashes
- **Cross-Sectional vs Time-Series**: Cross-sectional factors compare stocks to each other; time-series factors compare to own history
- **Factor Orthogonalization**: Isolate pure factor exposure by controlling for other factors; prevents unintended bets

## Core Factor Framework

### Classic Factors (Fama-French Extended)

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
import pandas as pd
from scipy import stats

@dataclass
class Factor:
    """Factor definition"""
    name: str
    long_description: str
    construction: str  # How to build long-short portfolio
    expected_premium: float  # Historical annual premium

# Classic Fama-French + Extensions
FACTORS = {
    'MKT': Factor('Market', 'Equity risk premium', 'Long stocks, short risk-free', 0.06),
    'SMB': Factor('Size', 'Small minus Big', 'Long small cap, short large cap', 0.02),
    'HML': Factor('Value', 'High minus Low B/M', 'Long high B/P, short low B/P', 0.03),
    'RMW': Factor('Profitability', 'Robust minus Weak', 'Long high ROE, short low ROE', 0.03),
    'CMA': Factor('Investment', 'Conservative minus Aggressive', 'Long low capex, short high capex', 0.02),
    'MOM': Factor('Momentum', '12-1 month returns', 'Long winners, short losers', 0.04),
    'QMJ': Factor('Quality', 'Quality minus Junk', 'Long quality, short junk', 0.03),
    'BAB': Factor('Betting Against Beta', 'Low beta outperformance', 'Long low beta, short high beta', 0.01),
    'LIQ': Factor('Liquidity', 'Illiquidity premium', 'Long illiquid, short liquid', 0.02),
}
```

### Factor Construction

```python
class FactorConstructor:
    """Build factor portfolios from stock universe"""

    def __init__(self, universe: pd.DataFrame):
        """
        universe: DataFrame with columns [date, ticker, market_cap, returns, ...]
        """
        self.universe = universe

    def construct_factor(
        self,
        characteristic: str,
        date: str,
        long_quantile: float = 0.7,
        short_quantile: float = 0.3,
        weighting: str = 'equal'  # 'equal' or 'value'
    ) -> Dict[str, float]:
        """
        Construct long-short factor portfolio

        Returns dict of {ticker: weight} for the factor portfolio
        """
        data = self.universe[self.universe['date'] == date].copy()

        # Rank stocks by characteristic
        data['rank'] = data[characteristic].rank(pct=True)

        # Long top quantile
        long_stocks = data[data['rank'] >= long_quantile]['ticker'].tolist()

        # Short bottom quantile
        short_stocks = data[data['rank'] <= short_quantile]['ticker'].tolist()

        # Calculate weights
        if weighting == 'equal':
            long_weight = 1.0 / len(long_stocks) if long_stocks else 0
            short_weight = -1.0 / len(short_stocks) if short_stocks else 0
        else:  # value-weighted
            long_caps = data[data['ticker'].isin(long_stocks)]['market_cap']
            short_caps = data[data['ticker'].isin(short_stocks)]['market_cap']
            long_weight = long_caps / long_caps.sum()
            short_weight = -short_caps / short_caps.sum()

        weights = {}
        for ticker in long_stocks:
            weights[ticker] = long_weight if weighting == 'equal' else long_weight[ticker]
        for ticker in short_stocks:
            weights[ticker] = short_weight if weighting == 'equal' else short_weight[ticker]

        return weights

    def momentum_factor(self, date: str, lookback: int = 12, skip: int = 1) -> Dict[str, float]:
        """
        Momentum: 12-1 month returns (skip most recent month)

        Classic momentum skips last month due to short-term reversal
        """
        # Calculate 12-month return excluding last month
        # ... implementation depends on data structure
        pass

    def value_factor(self, date: str, metric: str = 'book_to_market') -> Dict[str, float]:
        """
        Value: High book-to-market (or earnings yield, etc.)
        """
        return self.construct_factor(metric, date)

    def quality_factor(self, date: str) -> Dict[str, float]:
        """
        Quality: Composite of profitability, growth, safety

        QMJ = z(Profitability) + z(Growth) + z(Safety)
        """
        data = self.universe[self.universe['date'] == date].copy()

        # Profitability: ROE, ROA, gross margin
        data['prof_z'] = stats.zscore(data['roe'])

        # Growth: 5-year earnings growth
        data['growth_z'] = stats.zscore(data['earnings_growth_5y'])

        # Safety: Low leverage, low volatility
        data['safety_z'] = -stats.zscore(data['debt_to_equity'])

        # Composite quality score
        data['quality'] = data['prof_z'] + data['growth_z'] + data['safety_z']

        return self.construct_factor('quality', date)
```

## Factor Model Regression

### Fama-French Regression

```python
class FactorModel:
    """Multi-factor model for return decomposition"""

    def __init__(self, factor_returns: pd.DataFrame):
        """
        factor_returns: DataFrame with columns [date, MKT, SMB, HML, ...]
        """
        self.factor_returns = factor_returns

    def regress_returns(
        self,
        portfolio_returns: pd.Series,
        factors: List[str] = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'MOM']
    ) -> dict:
        """
        Regress portfolio returns on factor returns

        R_p - R_f = alpha + sum(beta_i * F_i) + epsilon

        Returns:
            alpha, betas, R², t-stats
        """
        import statsmodels.api as sm

        # Align dates
        aligned = pd.merge(
            portfolio_returns.to_frame('returns'),
            self.factor_returns[factors],
            left_index=True,
            right_index=True
        )

        y = aligned['returns']
        X = sm.add_constant(aligned[factors])

        model = sm.OLS(y, X).fit()

        return {
            'alpha': model.params['const'],
            'alpha_t': model.tvalues['const'],
            'alpha_pvalue': model.pvalues['const'],
            'betas': {f: model.params[f] for f in factors},
            'beta_t': {f: model.tvalues[f] for f in factors},
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'residual_vol': model.resid.std() * np.sqrt(252)
        }

    def factor_attribution(
        self,
        portfolio_returns: pd.Series,
        factors: List[str]
    ) -> pd.DataFrame:
        """
        Decompose portfolio returns into factor contributions

        Returns DataFrame with contribution from each factor
        """
        regression = self.regress_returns(portfolio_returns, factors)

        # Calculate factor contributions
        contributions = {}
        for factor in factors:
            beta = regression['betas'][factor]
            factor_return = self.factor_returns[factor].mean() * 252  # Annualized
            contributions[factor] = beta * factor_return

        contributions['alpha'] = regression['alpha'] * 252
        contributions['total'] = sum(contributions.values())

        return pd.Series(contributions)
```

### Information Ratio & Alpha Decay

```python
def calculate_information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series
) -> float:
    """
    Information Ratio = Alpha / Tracking Error

    IR > 0.5 is good, > 1.0 is excellent
    """
    excess_returns = portfolio_returns - benchmark_returns
    alpha = excess_returns.mean() * 252
    tracking_error = excess_returns.std() * np.sqrt(252)

    return alpha / tracking_error if tracking_error > 0 else 0

def analyze_alpha_decay(
    signal: pd.Series,
    forward_returns: pd.DataFrame,
    horizons: List[int] = [1, 5, 10, 21, 63]
) -> pd.DataFrame:
    """
    Analyze how signal alpha decays over time

    Critical for understanding signal half-life and rebalancing frequency
    """
    results = []

    for horizon in horizons:
        # Correlation between signal and forward returns
        fwd_ret = forward_returns[f'ret_{horizon}d']
        ic = signal.corr(fwd_ret)

        # Rank IC (Spearman)
        rank_ic = signal.corr(fwd_ret, method='spearman')

        results.append({
            'horizon_days': horizon,
            'ic': ic,
            'rank_ic': rank_ic,
            't_stat': ic * np.sqrt(len(signal) - 2) / np.sqrt(1 - ic**2)
        })

    return pd.DataFrame(results)
```

## Factor Timing

### Regime-Based Factor Allocation

```python
class FactorTimer:
    """Dynamic factor allocation based on regime"""

    def __init__(self, factor_returns: pd.DataFrame):
        self.factor_returns = factor_returns

    def momentum_regime(self, lookback: int = 12) -> Dict[str, float]:
        """
        Time factors based on their own momentum

        Overweight factors with positive recent returns
        """
        recent_returns = self.factor_returns.iloc[-lookback*21:].sum()

        weights = {}
        for factor in recent_returns.index:
            if recent_returns[factor] > 0:
                weights[factor] = 1.0
            else:
                weights[factor] = 0.0

        # Normalize
        total = sum(weights.values())
        return {f: w/total for f, w in weights.items()} if total > 0 else weights

    def value_spread_timing(self, factor: str, current_spread: float, historical_spreads: pd.Series) -> float:
        """
        Time value factor based on valuation spread

        When value spread is wide (cheap vs expensive gap is large),
        expected value premium is higher
        """
        percentile = stats.percentileofscore(historical_spreads, current_spread)

        # Overweight when spread is wide (high percentile)
        if percentile > 80:
            return 1.5  # 50% overweight
        elif percentile > 60:
            return 1.2
        elif percentile < 20:
            return 0.5  # Underweight when spread is narrow
        else:
            return 1.0

    def volatility_regime_allocation(self, vix: float) -> Dict[str, float]:
        """
        Adjust factor weights based on volatility regime

        Low vol: Favor momentum, growth
        High vol: Favor quality, low-vol
        """
        if vix < 15:
            return {'MOM': 0.3, 'HML': 0.2, 'SMB': 0.2, 'QMJ': 0.15, 'RMW': 0.15}
        elif vix < 25:
            return {'MOM': 0.2, 'HML': 0.2, 'SMB': 0.15, 'QMJ': 0.25, 'RMW': 0.2}
        else:  # High vol
            return {'MOM': 0.1, 'HML': 0.15, 'SMB': 0.1, 'QMJ': 0.35, 'RMW': 0.3}
```

## Cross-Sectional Analysis

```python
def cross_sectional_regression(
    returns: pd.Series,
    characteristics: pd.DataFrame,
    date: str
) -> dict:
    """
    Fama-MacBeth style cross-sectional regression

    R_i = gamma_0 + sum(gamma_k * X_ik) + epsilon_i

    Run cross-sectional regression each period, then average coefficients
    """
    import statsmodels.api as sm

    # Single period cross-sectional regression
    y = returns
    X = sm.add_constant(characteristics)

    model = sm.OLS(y, X).fit()

    return {
        'gammas': model.params.to_dict(),
        't_stats': model.tvalues.to_dict(),
        'r_squared': model.rsquared
    }

def fama_macbeth(
    panel_returns: pd.DataFrame,
    panel_characteristics: pd.DataFrame,
    characteristics: List[str]
) -> dict:
    """
    Full Fama-MacBeth procedure

    1. Run cross-sectional regression each period
    2. Average coefficients across time
    3. Calculate t-stats using time-series standard errors
    """
    dates = panel_returns.index.unique()
    all_gammas = []

    for date in dates:
        ret = panel_returns.loc[date]
        chars = panel_characteristics.loc[date][characteristics]

        result = cross_sectional_regression(ret, chars, date)
        all_gammas.append(result['gammas'])

    gammas_df = pd.DataFrame(all_gammas)

    return {
        'avg_gamma': gammas_df.mean().to_dict(),
        'gamma_t_stat': (gammas_df.mean() / gammas_df.std() * np.sqrt(len(dates))).to_dict(),
        'gamma_std': gammas_df.std().to_dict()
    }
```

## Production Implementation

```python
class FactorPortfolio:
    """Production multi-factor portfolio"""

    def __init__(
        self,
        factors: List[str],
        target_weights: Dict[str, float],
        rebalance_frequency: str = 'monthly'
    ):
        self.factors = factors
        self.target_weights = target_weights
        self.rebalance_frequency = rebalance_frequency

    def calculate_exposures(
        self,
        holdings: Dict[str, float],
        stock_betas: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate current factor exposures"""
        exposures = {f: 0.0 for f in self.factors}

        for ticker, weight in holdings.items():
            if ticker in stock_betas.index:
                for factor in self.factors:
                    exposures[factor] += weight * stock_betas.loc[ticker, factor]

        return exposures

    def target_deviation(
        self,
        current_exposures: Dict[str, float]
    ) -> Dict[str, float]:
        """How far are we from target exposures?"""
        return {
            f: self.target_weights.get(f, 0) - current_exposures.get(f, 0)
            for f in self.factors
        }
```

## Best Practices

1. **Use multiple factors**: Single factors have high volatility; combine for stability
2. **Neutral market exposure**: Long-short factors should be dollar-neutral and beta-neutral
3. **Transaction costs matter**: High-turnover factors (momentum) need cost-aware implementation
4. **Factor crowding**: Monitor when factors become crowded (everyone owns same stocks)
5. **Regime awareness**: Factor premiums vary by market regime

## Common Pitfalls

- **Data mining**: Finding "factors" that are just noise in historical data
- **Ignoring costs**: Momentum factor turnover can be 200%+ annually
- **Static allocation**: Not adjusting to factor valuations or regimes
- **Survivorship bias**: Testing only on stocks that still exist
- **Factor decay**: Alpha from public factors decays as more capital chases them

---

**Skill Type**: Finance - Factor Investing
**Complexity**: Advanced
**Typical Usage**: Systematic portfolio construction, alpha research
