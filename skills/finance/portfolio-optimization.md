---
name: portfolio-optimization
description: Load when user needs mean-variance optimization, risk parity, Black-Litterman model, efficient frontier, portfolio construction, or asset allocation strategies
trigger_keywords: [portfolio optimization, mean-variance, markowitz, efficient frontier, risk parity, black-litterman, asset allocation, portfolio construction, covariance matrix, correlation matrix, portfolio weights, maximum sharpe, minimum variance, risk budgeting]
---

# Portfolio Optimization Skill

Production-grade portfolio construction using Modern Portfolio Theory, risk parity, and Bayesian approaches (Black-Litterman) for systematic asset allocation.

## Core Concepts

### Optimization Approaches

**Mean-Variance Optimization (Markowitz)**:
- Maximize expected return for given risk level
- Minimize variance for given return target
- Find maximum Sharpe ratio portfolio
- Foundation of Modern Portfolio Theory (1952)

**Risk Parity**:
- Equal risk contribution from each asset
- More robust to estimation error than MVO
- Popular in institutional portfolios

**Black-Litterman**:
- Bayesian framework combining market equilibrium with investor views
- Reduces extreme weights from pure MVO
- More intuitive portfolio construction

### Key Metrics

```yaml
Returns:
  - Expected Return: Weighted average of asset returns
  - Portfolio Return: Sum of (weight × asset return)

Risk:
  - Portfolio Variance: w'Σw (weights × covariance × weights)
  - Portfolio Volatility: sqrt(variance)
  - Tracking Error: Volatility vs benchmark

Risk-Adjusted:
  - Sharpe Ratio: (Return - Risk-Free) / Volatility
  - Sortino Ratio: (Return - Risk-Free) / Downside Volatility
  - Information Ratio: Alpha / Tracking Error
```

## Implementation Patterns

### 1. Covariance Estimation

**Sample Covariance** (Basic)
```python
import numpy as np
import pandas as pd
from typing import Tuple

def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Sample covariance matrix

    Args:
        returns: DataFrame of asset returns (rows=time, columns=assets)

    Returns:
        Covariance matrix
    """
    return returns.cov()
```

**Ledoit-Wolf Shrinkage** (Production)
```python
from sklearn.covariance import LedoitWolf

def shrunk_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Ledoit-Wolf shrinkage covariance estimator
    More stable for high-dimensional portfolios

    Shrinks sample covariance toward structured estimator
    to reduce estimation error
    """
    lw = LedoitWolf().fit(returns)
    cov_matrix = pd.DataFrame(
        lw.covariance_,
        index=returns.columns,
        columns=returns.columns
    )
    return cov_matrix

def exponential_covariance(
    returns: pd.DataFrame,
    halflife: int = 63  # ~3 months for daily data
) -> pd.DataFrame:
    """
    Exponentially-weighted covariance matrix
    Gives more weight to recent observations
    """
    return returns.ewm(halflife=halflife).cov().iloc[-len(returns.columns):]
```

### 2. Mean-Variance Optimization

**Maximum Sharpe Ratio Portfolio**
```python
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Optional

@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    asset_names: list

def portfolio_performance(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> Tuple[float, float]:
    """Calculate portfolio return and volatility"""
    portfolio_return = np.dot(weights, expected_returns)
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_return, portfolio_vol

def negative_sharpe(
    weights: np.ndarray,
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> float:
    """Negative Sharpe ratio (for minimization)"""
    ret, vol = portfolio_performance(weights, expected_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol

def max_sharpe_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.02,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    allow_short: bool = False
) -> OptimizationResult:
    """
    Find portfolio that maximizes Sharpe ratio

    Args:
        expected_returns: Expected returns per asset
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate (annualized)
        weight_bounds: Min/max weight per asset
        allow_short: Allow negative weights

    Returns:
        OptimizationResult with optimal weights
    """
    n_assets = len(expected_returns)

    # Initial guess: equal weights
    init_weights = np.array([1.0 / n_assets] * n_assets)

    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
    ]

    # Bounds
    if allow_short:
        bounds = tuple((-1.0, 1.0) for _ in range(n_assets))
    else:
        bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(n_assets))

    # Optimize
    result = minimize(
        negative_sharpe,
        init_weights,
        args=(expected_returns.values, cov_matrix.values, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    ret, vol = portfolio_performance(
        optimal_weights, expected_returns.values, cov_matrix.values
    )
    sharpe = (ret - risk_free_rate) / vol

    return OptimizationResult(
        weights=optimal_weights,
        expected_return=ret,
        volatility=vol,
        sharpe_ratio=sharpe,
        asset_names=list(expected_returns.index)
    )
```

**Minimum Variance Portfolio**
```python
def min_variance_portfolio(
    cov_matrix: pd.DataFrame,
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
) -> OptimizationResult:
    """
    Find minimum variance portfolio
    Useful when return estimates are unreliable
    """
    n_assets = len(cov_matrix)
    init_weights = np.array([1.0 / n_assets] * n_assets)

    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix.values, weights))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((weight_bounds[0], weight_bounds[1]) for _ in range(n_assets))

    result = minimize(
        portfolio_variance,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    vol = np.sqrt(portfolio_variance(optimal_weights))

    return OptimizationResult(
        weights=optimal_weights,
        expected_return=0.0,  # Not calculated
        volatility=vol,
        sharpe_ratio=0.0,
        asset_names=list(cov_matrix.columns)
    )
```

### 3. Efficient Frontier

```python
def efficient_frontier(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    n_points: int = 50,
    risk_free_rate: float = 0.02
) -> pd.DataFrame:
    """
    Calculate efficient frontier points

    Returns:
        DataFrame with columns: return, volatility, sharpe, weights
    """
    # Find return range
    min_var = min_variance_portfolio(cov_matrix)
    max_sharpe = max_sharpe_portfolio(expected_returns, cov_matrix, risk_free_rate)

    # Use individual asset returns for range
    min_ret = expected_returns.min()
    max_ret = expected_returns.max()

    target_returns = np.linspace(min_ret, max_ret, n_points)

    results = []
    n_assets = len(expected_returns)

    for target in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w, t=target: np.dot(w, expected_returns.values) - t}
        ]

        bounds = tuple((0.0, 1.0) for _ in range(n_assets))

        result = minimize(
            lambda w: np.dot(w.T, np.dot(cov_matrix.values, w)),
            np.array([1.0 / n_assets] * n_assets),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if result.success:
            weights = result.x
            ret = np.dot(weights, expected_returns.values)
            vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0

            results.append({
                'return': ret,
                'volatility': vol,
                'sharpe': sharpe,
                'weights': weights
            })

    return pd.DataFrame(results)
```

### 4. Risk Parity

```python
def risk_parity_portfolio(
    cov_matrix: pd.DataFrame,
    risk_budget: Optional[np.ndarray] = None
) -> OptimizationResult:
    """
    Risk parity portfolio - equal risk contribution from each asset

    Args:
        cov_matrix: Covariance matrix
        risk_budget: Target risk contribution per asset (default: equal)

    Risk contribution formula:
        RC_i = w_i * (Σw)_i / sqrt(w'Σw)

    For equal risk contribution:
        RC_i = 1/n for all i
    """
    n_assets = len(cov_matrix)

    if risk_budget is None:
        risk_budget = np.array([1.0 / n_assets] * n_assets)

    def risk_contribution(weights):
        """Calculate risk contribution for each asset"""
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
        marginal_contrib = np.dot(cov_matrix.values, weights) / portfolio_vol
        risk_contrib = weights * marginal_contrib
        return risk_contrib / risk_contrib.sum()

    def objective(weights):
        """Minimize squared difference from target risk budget"""
        rc = risk_contribution(weights)
        return np.sum((rc - risk_budget) ** 2)

    # Constraints and bounds
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0.01, 1.0) for _ in range(n_assets))  # Min 1% per asset

    # Initial guess
    init_weights = np.array([1.0 / n_assets] * n_assets)

    result = minimize(
        objective,
        init_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    vol = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix.values, optimal_weights)))

    return OptimizationResult(
        weights=optimal_weights,
        expected_return=0.0,
        volatility=vol,
        sharpe_ratio=0.0,
        asset_names=list(cov_matrix.columns)
    )
```

### 5. Black-Litterman Model

```python
def black_litterman(
    market_caps: pd.Series,
    cov_matrix: pd.DataFrame,
    views: pd.DataFrame,
    view_confidences: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05,
    risk_free_rate: float = 0.02
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Black-Litterman model for portfolio optimization

    Combines market equilibrium with investor views using Bayesian updating

    Args:
        market_caps: Market capitalization per asset
        cov_matrix: Covariance matrix
        views: Pick matrix P (each row is a view)
        view_confidences: Confidence in each view (0-1)
        risk_aversion: Risk aversion parameter (typical: 2-4)
        tau: Scaling factor for uncertainty in equilibrium (typical: 0.025-0.05)
        risk_free_rate: Risk-free rate

    Returns:
        (posterior_returns, posterior_covariance)

    Black-Litterman Formula:
        π = δΣw_mkt  (equilibrium excess returns)
        E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 × [(τΣ)^-1π + P'Ω^-1Q]
    """
    # Market weights from market caps
    market_weights = market_caps / market_caps.sum()

    # Implied equilibrium returns (reverse optimization)
    # π = δΣw_mkt
    equilibrium_returns = risk_aversion * np.dot(cov_matrix.values, market_weights.values)

    # Uncertainty in views (Ω = diag(P × τΣ × P' × (1/confidence - 1)))
    # Simplified: Ω = diag(view_variance / confidence)
    P = views.values
    n_views = len(view_confidences)

    # View uncertainty matrix
    omega = np.diag(
        [np.dot(P[i], np.dot(cov_matrix.values * tau, P[i])) / view_confidences[i]
         for i in range(n_views)]
    )

    # Prior precision
    tau_sigma_inv = np.linalg.inv(tau * cov_matrix.values)

    # Posterior precision
    posterior_precision = tau_sigma_inv + np.dot(P.T, np.dot(np.linalg.inv(omega), P))

    # Posterior covariance
    posterior_cov = np.linalg.inv(posterior_precision)

    # View returns (Q) - assumed from views DataFrame index
    Q = views.index.values.astype(float)

    # Posterior returns
    posterior_returns = np.dot(
        posterior_cov,
        np.dot(tau_sigma_inv, equilibrium_returns) +
        np.dot(P.T, np.dot(np.linalg.inv(omega), Q))
    )

    return (
        pd.Series(posterior_returns, index=cov_matrix.columns),
        pd.DataFrame(posterior_cov, index=cov_matrix.columns, columns=cov_matrix.columns)
    )
```

## Production-Ready Portfolio Optimizer

```python
from dataclasses import dataclass
from typing import Literal, Optional, Dict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

@dataclass
class PortfolioConfig:
    """Configuration for portfolio optimization"""
    method: Literal['max_sharpe', 'min_variance', 'risk_parity', 'target_return'] = 'max_sharpe'
    risk_free_rate: float = 0.02
    target_return: Optional[float] = None
    min_weight: float = 0.0
    max_weight: float = 1.0
    allow_short: bool = False
    covariance_method: Literal['sample', 'shrinkage', 'exponential'] = 'shrinkage'
    halflife: int = 63  # For exponential covariance

class PortfolioOptimizer:
    """Production portfolio optimizer with multiple methods"""

    def __init__(self, returns: pd.DataFrame, config: PortfolioConfig = None):
        """
        Args:
            returns: DataFrame of asset returns (rows=dates, columns=assets)
            config: Optimization configuration
        """
        self.returns = returns
        self.config = config or PortfolioConfig()
        self.assets = list(returns.columns)
        self.n_assets = len(self.assets)

        # Calculate inputs
        self.expected_returns = self._estimate_returns()
        self.cov_matrix = self._estimate_covariance()

    def _estimate_returns(self) -> pd.Series:
        """Estimate expected returns (annualized)"""
        return self.returns.mean() * 252  # Annualize daily returns

    def _estimate_covariance(self) -> pd.DataFrame:
        """Estimate covariance matrix based on config"""
        if self.config.covariance_method == 'sample':
            cov = self.returns.cov() * 252
        elif self.config.covariance_method == 'shrinkage':
            lw = LedoitWolf().fit(self.returns)
            cov = pd.DataFrame(
                lw.covariance_ * 252,
                index=self.assets,
                columns=self.assets
            )
        else:  # exponential
            cov = self.returns.ewm(halflife=self.config.halflife).cov()
            cov = cov.iloc[-self.n_assets:] * 252

        return cov

    def optimize(self) -> OptimizationResult:
        """Run optimization based on config"""
        if self.config.method == 'max_sharpe':
            return self._max_sharpe()
        elif self.config.method == 'min_variance':
            return self._min_variance()
        elif self.config.method == 'risk_parity':
            return self._risk_parity()
        elif self.config.method == 'target_return':
            return self._target_return()
        else:
            raise ValueError(f"Unknown method: {self.config.method}")

    def _max_sharpe(self) -> OptimizationResult:
        """Maximum Sharpe ratio optimization"""
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        def neg_sharpe(w):
            ret = np.dot(w, self.expected_returns.values)
            vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix.values, w)))
            return -(ret - self.config.risk_free_rate) / vol

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = self._get_bounds()

        result = minimize(neg_sharpe, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return self._build_result(result.x)

    def _min_variance(self) -> OptimizationResult:
        """Minimum variance optimization"""
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        def variance(w):
            return np.dot(w.T, np.dot(self.cov_matrix.values, w))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = self._get_bounds()

        result = minimize(variance, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return self._build_result(result.x)

    def _risk_parity(self) -> OptimizationResult:
        """Risk parity optimization"""
        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)
        target_rc = np.array([1.0 / self.n_assets] * self.n_assets)

        def risk_contrib_diff(w):
            vol = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix.values, w)))
            marginal = np.dot(self.cov_matrix.values, w) / vol
            rc = w * marginal / vol
            return np.sum((rc - target_rc) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0.01, 1.0) for _ in range(self.n_assets))

        result = minimize(risk_contrib_diff, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return self._build_result(result.x)

    def _target_return(self) -> OptimizationResult:
        """Optimize for target return with minimum variance"""
        if self.config.target_return is None:
            raise ValueError("target_return must be set in config")

        init_weights = np.array([1.0 / self.n_assets] * self.n_assets)

        def variance(w):
            return np.dot(w.T, np.dot(self.cov_matrix.values, w))

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: np.dot(w, self.expected_returns.values) - self.config.target_return}
        ]
        bounds = self._get_bounds()

        result = minimize(variance, init_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)

        return self._build_result(result.x)

    def _get_bounds(self):
        """Get weight bounds based on config"""
        if self.config.allow_short:
            return tuple((-1.0, 1.0) for _ in range(self.n_assets))
        return tuple((self.config.min_weight, self.config.max_weight)
                    for _ in range(self.n_assets))

    def _build_result(self, weights: np.ndarray) -> OptimizationResult:
        """Build result object from weights"""
        ret = np.dot(weights, self.expected_returns.values)
        vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix.values, weights)))
        sharpe = (ret - self.config.risk_free_rate) / vol if vol > 0 else 0

        return OptimizationResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            asset_names=self.assets
        )

    def get_efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Calculate efficient frontier"""
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        targets = np.linspace(min_ret, max_ret, n_points)

        results = []
        for target in targets:
            self.config.target_return = target
            self.config.method = 'target_return'
            try:
                result = self.optimize()
                results.append({
                    'return': result.expected_return,
                    'volatility': result.volatility,
                    'sharpe': result.sharpe_ratio
                })
            except:
                continue

        return pd.DataFrame(results)

# Usage Example
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_days = 252 * 3  # 3 years
    assets = ['SPY', 'AGG', 'GLD', 'VNQ', 'EEM']

    returns = pd.DataFrame(
        np.random.randn(n_days, len(assets)) * 0.01 + 0.0003,
        columns=assets
    )

    # Max Sharpe optimization
    config = PortfolioConfig(method='max_sharpe', risk_free_rate=0.02)
    optimizer = PortfolioOptimizer(returns, config)
    result = optimizer.optimize()

    print("Maximum Sharpe Portfolio:")
    for asset, weight in zip(result.asset_names, result.weights):
        print(f"  {asset}: {weight:.1%}")
    print(f"\nExpected Return: {result.expected_return:.1%}")
    print(f"Volatility: {result.volatility:.1%}")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")

    # Risk Parity
    config.method = 'risk_parity'
    result_rp = optimizer.optimize()

    print("\nRisk Parity Portfolio:")
    for asset, weight in zip(result_rp.asset_names, result_rp.weights):
        print(f"  {asset}: {weight:.1%}")
```

## Constraints and Extensions

### Sector/Factor Constraints

```python
def optimize_with_constraints(
    optimizer: PortfolioOptimizer,
    sector_map: Dict[str, str],
    max_sector_weight: float = 0.30,
    min_assets: int = 5
) -> OptimizationResult:
    """
    Optimization with sector and diversification constraints
    """
    sectors = list(set(sector_map.values()))

    additional_constraints = []

    # Sector constraints
    for sector in sectors:
        sector_assets = [i for i, a in enumerate(optimizer.assets)
                       if sector_map.get(a) == sector]
        additional_constraints.append({
            'type': 'ineq',
            'fun': lambda w, idx=sector_assets: max_sector_weight - sum(w[i] for i in idx)
        })

    # Minimum number of assets
    additional_constraints.append({
        'type': 'ineq',
        'fun': lambda w: sum(1 for wi in w if wi > 0.01) - min_assets
    })

    # Run optimization with extended constraints
    # (Implementation would extend optimizer.optimize())
    pass
```

## Best Practices

1. **Use shrinkage estimators** for covariance (Ledoit-Wolf) in production
2. **Regularize expected returns** - they're notoriously hard to estimate
3. **Add constraints** to prevent extreme weights and improve robustness
4. **Rebalance periodically** but consider transaction costs
5. **Validate with walk-forward analysis** before live deployment
6. **Consider estimation error** - wider confidence intervals = more conservative

## Common Pitfalls

❌ **Using sample covariance** with many assets (estimation error explodes)
✅ Use Ledoit-Wolf shrinkage or factor models

❌ **Trusting expected return estimates** without skepticism
✅ Use equilibrium returns (Black-Litterman) or risk-based methods (risk parity)

❌ **No constraints** on individual weights
✅ Set min/max bounds to prevent concentration

❌ **Optimizing once** and forgetting
✅ Implement systematic rebalancing with cost awareness

❌ **Ignoring transaction costs** in optimization
✅ Add turnover constraints or regularization

## Quality Standards

- **Numerical Stability**: Condition number of covariance < 1e6
- **Weight Bounds**: All weights within specified bounds ± 1e-6
- **Convergence**: Optimization converges within 1000 iterations
- **Validation**: Backtested Sharpe within 0.3 of expected
- **Diversification**: Effective N > 3 for production portfolios

---

**Skill Type**: Finance - Portfolio Construction
**Complexity**: Complex
**Typical Usage**: Activated when portfolio-manager or quantitative-analyst needs asset allocation
**Performance**: Optimization completes in <100ms for 50 assets
