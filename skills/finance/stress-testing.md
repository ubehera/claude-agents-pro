---
name: stress-testing
description: Load when user needs tail risk analysis, stress testing, CVaR, scenario analysis, or correlation breakdown modeling. Covers extreme event risk management.
trigger_keywords: [stress test, tail risk, cvar, expected shortfall, scenario analysis, correlation breakdown, black swan, fat tails, extreme events, crisis, drawdown, worst case, historical simulation, monte carlo stress, var breach]
---

# Stress Testing & Tail Risk Skill

Tail risk analysis, scenario-based stress testing, and extreme event modeling for capital preservation.

## Core Concepts

- **VaR Limitations**: VaR only tells you the threshold; says nothing about losses beyond that threshold (use CVaR instead)
- **Fat Tails**: Financial returns have fatter tails than normal distribution; 3-sigma events happen more than expected
- **Correlation Breakdown**: In crises, correlations spike toward 1.0; diversification fails when you need it most
- **Tail Dependence**: Assets that seem uncorrelated in normal times can crash together in stress scenarios
- **Scenario vs Historical**: Historical simulations miss unprecedented events; scenario analysis captures "what if" risks

## Tail Risk Metrics

### Expected Shortfall (CVaR)

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict

def calculate_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = 'historical'
) -> float:
    """
    Value at Risk

    "95% of days, we lose less than X%"
    But says NOTHING about the 5% worst days
    """
    if method == 'historical':
        return np.percentile(returns, (1 - confidence) * 100)
    elif method == 'parametric':
        return returns.mean() + stats.norm.ppf(1 - confidence) * returns.std()
    elif method == 'cornish_fisher':
        # Adjusts for skewness and kurtosis
        z = stats.norm.ppf(1 - confidence)
        s = stats.skew(returns)
        k = stats.kurtosis(returns)
        z_cf = (z + (z**2 - 1) * s / 6 +
                (z**3 - 3*z) * (k - 3) / 24 -
                (2*z**3 - 5*z) * s**2 / 36)
        return returns.mean() + z_cf * returns.std()

def calculate_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """
    Conditional VaR (Expected Shortfall)

    "Given we're in the worst 5%, what's the average loss?"

    CVaR is always worse than VaR - captures tail severity
    """
    var = calculate_var(returns, confidence)
    tail_returns = returns[returns <= var]
    return tail_returns.mean()

def tail_risk_metrics(returns: pd.Series) -> dict:
    """
    Comprehensive tail risk analysis
    """
    return {
        'var_95': calculate_var(returns, 0.95),
        'var_99': calculate_var(returns, 0.99),
        'cvar_95': calculate_cvar(returns, 0.95),
        'cvar_99': calculate_cvar(returns, 0.99),
        'max_drawdown': calculate_max_drawdown(returns),
        'skewness': stats.skew(returns),
        'kurtosis': stats.kurtosis(returns),  # Excess kurtosis
        'tail_ratio': abs(np.percentile(returns, 5) / np.percentile(returns, 95))
    }

def calculate_max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough decline"""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

### Fat Tail Analysis

```python
def fit_tail_distribution(
    returns: pd.Series,
    tail: str = 'left',
    threshold_percentile: float = 5
) -> dict:
    """
    Fit Generalized Pareto Distribution to tail

    GPD is theoretically justified for extreme values
    """
    if tail == 'left':
        threshold = np.percentile(returns, threshold_percentile)
        exceedances = threshold - returns[returns < threshold]
    else:
        threshold = np.percentile(returns, 100 - threshold_percentile)
        exceedances = returns[returns > threshold] - threshold

    # Fit GPD
    from scipy.stats import genpareto
    params = genpareto.fit(exceedances)

    return {
        'shape': params[0],  # xi - tail heaviness
        'scale': params[2],  # sigma
        'threshold': threshold,
        'n_exceedances': len(exceedances),
        'interpretation': 'heavy_tail' if params[0] > 0 else 'bounded_tail'
    }

def estimate_extreme_quantile(
    returns: pd.Series,
    probability: float = 0.001,  # 1 in 1000 day event
    method: str = 'gpd'
) -> float:
    """
    Estimate extreme loss for very rare events

    Standard VaR fails for 99.9% because not enough data
    GPD extrapolates from observed tail
    """
    if method == 'gpd':
        tail_fit = fit_tail_distribution(returns)
        from scipy.stats import genpareto

        xi = tail_fit['shape']
        sigma = tail_fit['scale']
        threshold = tail_fit['threshold']
        n = len(returns)
        n_exceed = tail_fit['n_exceedances']

        # Extrapolate using GPD
        exceed_prob = n_exceed / n
        if xi != 0:
            extreme_loss = threshold - (sigma / xi) * (
                (probability / exceed_prob) ** (-xi) - 1
            )
        else:
            extreme_loss = threshold - sigma * np.log(probability / exceed_prob)

        return extreme_loss
    else:
        # Simple historical
        return np.percentile(returns, probability * 100)
```

## Scenario Analysis

### Historical Scenarios

```python
HISTORICAL_SCENARIOS = {
    'black_monday_1987': {
        'name': 'Black Monday 1987',
        'date': '1987-10-19',
        'equity_shock': -0.225,  # -22.5% single day
        'vol_spike': 2.5,        # VIX multiplier
        'duration_days': 1
    },
    'asian_crisis_1997': {
        'name': 'Asian Financial Crisis',
        'date': '1997-10-27',
        'equity_shock': -0.07,
        'em_shock': -0.30,
        'duration_days': 90
    },
    'dot_com_crash_2000': {
        'name': 'Dot-Com Crash',
        'date': '2000-03-10',
        'equity_shock': -0.49,
        'nasdaq_shock': -0.78,
        'duration_days': 730
    },
    'gfc_2008': {
        'name': 'Global Financial Crisis',
        'date': '2008-09-15',
        'equity_shock': -0.57,
        'credit_spread_bps': 600,
        'vol_spike': 3.0,
        'correlation_spike': 0.90,
        'duration_days': 365
    },
    'covid_crash_2020': {
        'name': 'COVID-19 Crash',
        'date': '2020-03-16',
        'equity_shock': -0.34,
        'vol_spike': 4.0,
        'recovery_days': 120,
        'duration_days': 33
    },
    'inflation_2022': {
        'name': '2022 Inflation Shock',
        'date': '2022-01-03',
        'equity_shock': -0.25,
        'bond_shock': -0.20,  # Bonds down too!
        'correlation_break': True,
        'duration_days': 280
    }
}

class ScenarioAnalyzer:
    """Apply historical scenarios to portfolio"""

    def __init__(self, portfolio: Dict[str, float]):
        """
        portfolio: {asset: weight} dict
        """
        self.portfolio = portfolio

    def apply_scenario(
        self,
        scenario: dict,
        asset_betas: Dict[str, float] = None
    ) -> dict:
        """
        Calculate portfolio impact under scenario
        """
        if asset_betas is None:
            asset_betas = {asset: 1.0 for asset in self.portfolio}

        total_impact = 0
        asset_impacts = {}

        for asset, weight in self.portfolio.items():
            beta = asset_betas.get(asset, 1.0)
            impact = weight * beta * scenario.get('equity_shock', 0)
            asset_impacts[asset] = impact
            total_impact += impact

        return {
            'scenario': scenario['name'],
            'portfolio_impact': total_impact,
            'asset_impacts': asset_impacts,
            'worst_asset': min(asset_impacts, key=asset_impacts.get),
            'recovery_estimate_days': scenario.get('recovery_days', scenario.get('duration_days', 0))
        }

    def run_all_scenarios(self) -> pd.DataFrame:
        """Run portfolio through all historical scenarios"""
        results = []
        for name, scenario in HISTORICAL_SCENARIOS.items():
            result = self.apply_scenario(scenario)
            results.append({
                'scenario': name,
                'impact': result['portfolio_impact'],
                'duration': scenario.get('duration_days', 0)
            })
        return pd.DataFrame(results).sort_values('impact')
```

### Custom Stress Scenarios

```python
def create_custom_scenario(
    equity_shock: float,
    bond_shock: float = 0,
    vol_multiplier: float = 1.5,
    correlation_override: float = None,
    credit_spread_bps: float = 0
) -> dict:
    """
    Build custom stress scenario
    """
    return {
        'name': 'Custom Scenario',
        'equity_shock': equity_shock,
        'bond_shock': bond_shock,
        'vol_spike': vol_multiplier,
        'correlation_spike': correlation_override,
        'credit_spread_bps': credit_spread_bps
    }

def sensitivity_grid(
    portfolio: Dict[str, float],
    equity_shocks: List[float] = [-0.10, -0.20, -0.30, -0.40, -0.50],
    vol_multipliers: List[float] = [1.0, 1.5, 2.0, 3.0]
) -> pd.DataFrame:
    """
    Create sensitivity grid for multiple shock combinations
    """
    analyzer = ScenarioAnalyzer(portfolio)
    results = []

    for eq_shock in equity_shocks:
        for vol_mult in vol_multipliers:
            scenario = create_custom_scenario(
                equity_shock=eq_shock,
                vol_multiplier=vol_mult
            )
            impact = analyzer.apply_scenario(scenario)
            results.append({
                'equity_shock': eq_shock,
                'vol_multiplier': vol_mult,
                'portfolio_impact': impact['portfolio_impact']
            })

    return pd.DataFrame(results).pivot(
        index='equity_shock',
        columns='vol_multiplier',
        values='portfolio_impact'
    )
```

## Correlation Breakdown

### Dynamic Correlation

```python
def rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window: int = 60
) -> pd.Series:
    """Rolling correlation between two assets"""
    return returns_a.rolling(window).corr(returns_b)

def correlation_during_stress(
    returns: pd.DataFrame,
    stress_threshold: float = -0.02  # 2% down day
) -> pd.DataFrame:
    """
    Calculate correlation matrix during stress vs normal

    Key insight: Correlations spike during stress
    """
    # Identify stress days (market down > threshold)
    market_returns = returns.mean(axis=1)
    stress_days = market_returns < stress_threshold
    normal_days = ~stress_days

    stress_corr = returns[stress_days].corr()
    normal_corr = returns[normal_days].corr()

    return {
        'stress_correlation': stress_corr,
        'normal_correlation': normal_corr,
        'correlation_increase': stress_corr - normal_corr,
        'stress_days_count': stress_days.sum(),
        'normal_days_count': normal_days.sum()
    }

def correlation_regime_detection(
    returns: pd.DataFrame,
    window: int = 60
) -> pd.DataFrame:
    """
    Detect correlation regimes using eigenvalue analysis

    High first eigenvalue = high correlation regime
    """
    results = []

    for i in range(window, len(returns)):
        window_returns = returns.iloc[i-window:i]
        corr_matrix = window_returns.corr()

        # Eigenvalue decomposition
        eigenvalues = np.linalg.eigvalsh(corr_matrix)
        first_eigenvalue = eigenvalues[-1]  # Largest

        # Proportion of variance explained by first PC
        variance_explained = first_eigenvalue / len(eigenvalues)

        results.append({
            'date': returns.index[i],
            'first_eigenvalue': first_eigenvalue,
            'variance_explained': variance_explained,
            'regime': 'high_corr' if variance_explained > 0.5 else 'normal'
        })

    return pd.DataFrame(results)
```

### Stressed Correlation Matrix

```python
def stressed_correlation_matrix(
    normal_corr: pd.DataFrame,
    stress_factor: float = 0.5  # Move correlations toward 1
) -> pd.DataFrame:
    """
    Generate stressed correlation matrix

    In crisis, correlations move toward 1 (or -1 for hedges)
    """
    stressed = normal_corr.copy()

    for i in range(len(stressed)):
        for j in range(len(stressed)):
            if i != j:
                current = normal_corr.iloc[i, j]
                # Move toward 1 (or -1 if already negative)
                target = 1.0 if current >= 0 else -1.0
                stressed.iloc[i, j] = current + stress_factor * (target - current)

    return stressed

def portfolio_vol_under_stress(
    weights: np.ndarray,
    normal_corr: pd.DataFrame,
    vols: np.ndarray,
    stress_factor: float = 0.5
) -> dict:
    """
    Compare portfolio vol under normal vs stressed correlations
    """
    stressed_corr = stressed_correlation_matrix(normal_corr, stress_factor)

    # Convert correlation to covariance
    def corr_to_cov(corr, vols):
        vol_matrix = np.outer(vols, vols)
        return corr * vol_matrix

    normal_cov = corr_to_cov(normal_corr.values, vols)
    stressed_cov = corr_to_cov(stressed_corr.values, vols)

    normal_port_var = weights @ normal_cov @ weights
    stressed_port_var = weights @ stressed_cov @ weights

    return {
        'normal_vol': np.sqrt(normal_port_var),
        'stressed_vol': np.sqrt(stressed_port_var),
        'vol_increase': np.sqrt(stressed_port_var) / np.sqrt(normal_port_var) - 1,
        'diversification_benefit_lost': 1 - (np.sqrt(stressed_port_var) - np.sqrt(normal_port_var)) / np.sqrt(normal_port_var)
    }
```

## Monte Carlo Stress Testing

```python
def monte_carlo_stress_test(
    returns: pd.DataFrame,
    n_simulations: int = 10000,
    horizon_days: int = 21,
    include_jumps: bool = True
) -> dict:
    """
    Monte Carlo simulation with fat tails and jumps
    """
    # Fit parameters
    means = returns.mean()
    cov = returns.cov()

    # Add jump component for fat tails
    if include_jumps:
        jump_prob = 0.02  # 2% chance of jump per day
        jump_size_mean = -0.05  # Jumps are typically negative
        jump_size_std = 0.03

    simulated_returns = []

    for _ in range(n_simulations):
        sim_path = np.zeros((horizon_days, len(returns.columns)))

        for t in range(horizon_days):
            # Normal returns
            daily_return = np.random.multivariate_normal(means, cov)

            # Add jumps
            if include_jumps and np.random.random() < jump_prob:
                jump = np.random.normal(jump_size_mean, jump_size_std, len(returns.columns))
                daily_return += jump

            sim_path[t] = daily_return

        # Calculate total return over horizon
        total_return = (1 + sim_path).prod(axis=0) - 1
        simulated_returns.append(total_return.mean())  # Portfolio return

    simulated_returns = np.array(simulated_returns)

    return {
        'mean': simulated_returns.mean(),
        'std': simulated_returns.std(),
        'var_95': np.percentile(simulated_returns, 5),
        'var_99': np.percentile(simulated_returns, 1),
        'cvar_95': simulated_returns[simulated_returns <= np.percentile(simulated_returns, 5)].mean(),
        'max_loss': simulated_returns.min(),
        'prob_loss_10pct': (simulated_returns < -0.10).mean(),
        'prob_loss_20pct': (simulated_returns < -0.20).mean()
    }
```

## Production Stress Testing Framework

```python
class StressTestFramework:
    """Production stress testing system"""

    def __init__(
        self,
        portfolio: Dict[str, float],
        returns_history: pd.DataFrame
    ):
        self.portfolio = portfolio
        self.returns = returns_history
        self.results = {}

    def run_full_stress_test(self) -> dict:
        """Run comprehensive stress test suite"""

        # 1. Historical scenarios
        scenario_analyzer = ScenarioAnalyzer(self.portfolio)
        self.results['historical_scenarios'] = scenario_analyzer.run_all_scenarios()

        # 2. Tail risk metrics
        portfolio_returns = (self.returns * pd.Series(self.portfolio)).sum(axis=1)
        self.results['tail_metrics'] = tail_risk_metrics(portfolio_returns)

        # 3. Correlation stress
        weights = np.array(list(self.portfolio.values()))
        vols = self.returns.std().values * np.sqrt(252)
        self.results['correlation_stress'] = portfolio_vol_under_stress(
            weights, self.returns.corr(), vols
        )

        # 4. Monte Carlo
        self.results['monte_carlo'] = monte_carlo_stress_test(self.returns)

        # 5. Summary
        self.results['summary'] = self._generate_summary()

        return self.results

    def _generate_summary(self) -> dict:
        """Generate executive summary"""
        return {
            'worst_historical_scenario': self.results['historical_scenarios'].iloc[0]['scenario'],
            'worst_historical_impact': self.results['historical_scenarios'].iloc[0]['impact'],
            'var_99': self.results['tail_metrics']['var_99'],
            'cvar_99': self.results['tail_metrics']['cvar_99'],
            'stressed_vol_increase': self.results['correlation_stress']['vol_increase'],
            'monte_carlo_worst': self.results['monte_carlo']['max_loss'],
            'recommendation': self._generate_recommendation()
        }

    def _generate_recommendation(self) -> str:
        """Generate risk recommendation"""
        cvar = self.results['tail_metrics']['cvar_99']
        if cvar < -0.20:
            return "HIGH RISK: Consider reducing position sizes or adding hedges"
        elif cvar < -0.10:
            return "MODERATE RISK: Monitor closely, consider tail hedges"
        else:
            return "ACCEPTABLE RISK: Within normal parameters"
```

## Best Practices

1. **CVaR over VaR**: VaR doesn't capture tail severity
2. **Multiple scenarios**: Don't rely on single worst-case
3. **Correlation stress**: Always test with elevated correlations
4. **Historical + hypothetical**: Use both real crises and custom scenarios
5. **Regular updates**: Run stress tests weekly minimum

## Common Pitfalls

- **Trusting VaR**: VaR says nothing about magnitude of tail losses
- **Static correlations**: Correlations spike in crisis
- **Insufficient history**: 10 years may not include relevant crisis
- **Ignoring liquidity**: Stressed markets have liquidity gaps
- **Single scenario reliance**: 2022 showed bonds and stocks can crash together

---

**Skill Type**: Finance - Stress Testing
**Complexity**: Advanced
**Typical Usage**: Risk management, capital preservation, regulatory compliance
