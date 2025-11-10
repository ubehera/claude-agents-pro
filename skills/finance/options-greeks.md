---
name: options-greeks
description: Load when user needs options Greeks (delta, gamma, theta, vega, rho), Black-Scholes pricing, implied volatility calculations, or options analysis
trigger_keywords: [greeks, delta, gamma, theta, vega, rho, black-scholes, implied volatility, option pricing, call, put, european option, american option, iv, options analysis]
---

# Options Greeks and Pricing Skill

Comprehensive options pricing and Greeks calculations using Black-Scholes model with numerical methods for implied volatility extraction.

## Core Concepts

### Options Greeks Overview

**First-Order Greeks** (Price Sensitivity):
- **Delta (Δ)**: Rate of change of option price with respect to underlying price (0-1 for calls, -1-0 for puts)
- **Vega (ν)**: Sensitivity to volatility changes (always positive for long options)
- **Theta (Θ)**: Time decay - change in option value per day (negative for long options)
- **Rho (ρ)**: Sensitivity to interest rate changes

**Second-Order Greeks** (Curvature):
- **Gamma (Γ)**: Rate of change of delta with respect to underlying price (convexity measure)

### Black-Scholes Framework

**Assumptions**:
- European-style options (no early exercise)
- Constant volatility and risk-free rate
- No dividends (or dividend yield incorporated)
- Lognormal stock price distribution
- No transaction costs

**When to Use**:
- ✅ European calls/puts
- ✅ Quick pricing estimates
- ✅ Greeks for hedging
- ⚠️ American options (use binomial trees instead)
- ⚠️ Exotic options (use Monte Carlo)

## Implementation Patterns

### Core Black-Scholes Functions

**1. d1 and d2 Calculation** (Foundation for all pricing)
```python
import numpy as np
from scipy.stats import norm
from typing import Literal

def calculate_d1_d2(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> tuple[float, float]:
    """
    Calculate d1 and d2 for Black-Scholes formula

    Args:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)

    Returns:
        (d1, d2)
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2
```

**2. Black-Scholes Pricing**
```python
def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Black-Scholes option pricing

    Returns:
        Option price
    """
    if T <= 0:
        # Handle expiration case
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price
```

**3. Delta Calculation**
```python
def calculate_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Delta - First derivative of option price with respect to stock price

    Call delta: 0 to 1
    Put delta: -1 to 0

    Interpretation:
    - Delta of 0.50 means option moves $0.50 for every $1 move in stock
    - At-the-money options typically have delta ≈ 0.50 (call) or -0.50 (put)
    """
    if T <= 0:
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1, _ = calculate_d1_d2(S, K, T, r, sigma)

    if option_type == 'call':
        return norm.cdf(d1)
    else:  # put
        return norm.cdf(d1) - 1
```

**4. Gamma Calculation**
```python
def calculate_gamma(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Gamma - Second derivative of option price with respect to stock price
    Same for calls and puts

    Interpretation:
    - Highest for at-the-money options
    - Measures convexity of option position
    - High gamma = delta changes rapidly
    """
    if T <= 0:
        return 0.0

    d1, _ = calculate_d1_d2(S, K, T, r, sigma)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma
```

**5. Theta Calculation**
```python
def calculate_theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Theta - Time decay (per calendar day)
    Negative for long options (time decay hurts holder)

    Returns theta per day (annual theta / 365)

    Interpretation:
    - Theta of -0.05 means option loses $0.05 in value per day
    - Accelerates as expiration approaches
    """
    if T <= 0:
        return 0.0

    d1, d2 = calculate_d1_d2(S, K, T, r, sigma)

    # First term (same for calls and puts)
    first_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option_type == 'call':
        second_term = -r * K * np.exp(-r * T) * norm.cdf(d2)
        theta_annual = first_term + second_term
    else:  # put
        second_term = r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta_annual = first_term + second_term

    # Convert to per-day theta
    return theta_annual / 365
```

**6. Vega Calculation**
```python
def calculate_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    """
    Vega - Sensitivity to volatility (per 1% change in volatility)
    Same for calls and puts
    Always positive for long options

    Returns vega per 1% volatility change

    Interpretation:
    - Vega of 0.20 means option gains $0.20 for 1% increase in volatility
    - Highest for at-the-money options
    """
    if T <= 0:
        return 0.0

    d1, _ = calculate_d1_d2(S, K, T, r, sigma)

    vega = S * norm.pdf(d1) * np.sqrt(T)

    # Return vega per 1% volatility change
    return vega / 100
```

**7. Rho Calculation**
```python
def calculate_rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal['call', 'put']
) -> float:
    """
    Rho - Sensitivity to interest rate (per 1% change in rate)

    Returns rho per 1% interest rate change

    Interpretation:
    - Rho of 0.30 means option gains $0.30 for 1% increase in rates
    - Less important for short-dated options
    """
    if T <= 0:
        return 0.0

    _, d2 = calculate_d1_d2(S, K, T, r, sigma)

    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    # Return rho per 1% rate change
    return rho / 100
```

## Implied Volatility Calculation

### Newton-Raphson Method

```python
def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal['call', 'put'],
    initial_guess: float = 0.25,
    max_iterations: int = 100,
    tolerance: float = 1e-6
) -> float:
    """
    Calculate implied volatility using Newton-Raphson method

    Args:
        market_price: Observed market price of option
        S, K, T, r: Black-Scholes parameters
        option_type: 'call' or 'put'
        initial_guess: Starting volatility (default 25%)
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        Implied volatility (annualized)

    Raises:
        ValueError: If convergence fails or inputs invalid
    """
    # Validation
    if market_price <= 0:
        raise ValueError("Market price must be positive")
    if T <= 0:
        raise ValueError("Time to expiration must be positive")

    # Check for intrinsic value violations
    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price < intrinsic:
        raise ValueError(f"Market price {market_price} below intrinsic value {intrinsic}")

    # Newton-Raphson iteration
    sigma = initial_guess

    for i in range(max_iterations):
        # Calculate price and vega at current sigma
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        vega = calculate_vega(S, K, T, r, sigma) * 100  # Undo per-1% scaling

        # Check for convergence
        price_diff = market_price - price
        if abs(price_diff) < tolerance:
            return sigma

        # Avoid division by zero
        if vega < 1e-10:
            raise ValueError("Vega too small - cannot converge")

        # Newton-Raphson update
        sigma = sigma + price_diff / vega

        # Keep sigma positive
        if sigma <= 0:
            sigma = initial_guess / 2

    raise ValueError(f"IV calculation did not converge after {max_iterations} iterations")
```

## Production-Ready Class Implementation

```python
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
from scipy.stats import norm

@dataclass
class OptionsGreeks:
    """Container for all Greeks"""
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    price: float

class OptionsAnalysis:
    """Complete options pricing and Greeks analysis"""

    @staticmethod
    def calculate_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal['call', 'put']
    ) -> dict[str, float]:
        """
        Calculate all Greeks and option price

        Returns:
            Dictionary with keys: price, delta, gamma, theta, vega, rho
        """
        # Price
        price = black_scholes_price(S, K, T, r, sigma, option_type)

        # Greeks
        delta = calculate_delta(S, K, T, r, sigma, option_type)
        gamma = calculate_gamma(S, K, T, r, sigma)
        theta = calculate_theta(S, K, T, r, sigma, option_type)
        vega = calculate_vega(S, K, T, r, sigma)
        rho = calculate_rho(S, K, T, r, sigma, option_type)

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: Literal['call', 'put'] = 'call',
        initial_guess: float = 0.25,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """Calculate implied volatility"""
        return implied_volatility(
            market_price, S, K, T, r, option_type,
            initial_guess, max_iterations, tolerance
        )

    @staticmethod
    def delta_neutral_hedge(
        option_delta: float,
        option_contracts: int,
        shares_per_contract: int = 100
    ) -> int:
        """
        Calculate stock shares needed for delta-neutral hedge

        Args:
            option_delta: Delta of option position
            option_contracts: Number of option contracts
            shares_per_contract: Shares per contract (default 100)

        Returns:
            Number of shares to short/long (negative = short)
        """
        total_delta = option_delta * option_contracts * shares_per_contract
        return -int(total_delta)

# Usage Example
if __name__ == "__main__":
    # Calculate Greeks for an at-the-money call
    greeks = OptionsAnalysis.calculate_greeks(
        S=100,      # Stock price
        K=100,      # Strike price
        T=0.25,     # 3 months to expiration
        r=0.05,     # 5% risk-free rate
        sigma=0.25, # 25% volatility
        option_type='call'
    )

    print("Option Greeks:")
    print(f"Price: ${greeks['price']:.2f}")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Theta: ${greeks['theta']:.4f} per day")
    print(f"Vega: ${greeks['vega']:.4f} per 1% vol")
    print(f"Rho: ${greeks['rho']:.4f} per 1% rate")

    # Calculate implied volatility
    market_price = 5.50
    iv = OptionsAnalysis.implied_volatility(
        market_price=market_price,
        S=100, K=105, T=0.25, r=0.05,
        option_type='call'
    )
    print(f"\nImplied Volatility: {iv:.2%}")

    # Delta hedging
    hedge_shares = OptionsAnalysis.delta_neutral_hedge(
        option_delta=greeks['delta'],
        option_contracts=10
    )
    print(f"\nDelta hedge: {'Short' if hedge_shares < 0 else 'Long'} {abs(hedge_shares)} shares")
```

## Advanced Patterns

### Implied Volatility Surface

```python
import pandas as pd

def calculate_iv_surface(
    market_data: pd.DataFrame,
    S: float,
    r: float
) -> pd.DataFrame:
    """
    Calculate IV surface from market option prices

    Args:
        market_data: DataFrame with columns [strike, expiration, price, type]
        S: Current stock price
        r: Risk-free rate

    Returns:
        DataFrame with IV surface
    """
    results = []

    for _, row in market_data.iterrows():
        try:
            iv = OptionsAnalysis.implied_volatility(
                market_price=row['price'],
                S=S,
                K=row['strike'],
                T=row['expiration'],
                r=r,
                option_type=row['type']
            )
            results.append({
                'strike': row['strike'],
                'expiration': row['expiration'],
                'type': row['type'],
                'iv': iv,
                'moneyness': row['strike'] / S
            })
        except ValueError:
            # Skip options where IV calculation fails
            continue

    return pd.DataFrame(results)
```

### Greeks Hedging Portfolio

```python
def calculate_portfolio_greeks(
    positions: list[dict]
) -> dict[str, float]:
    """
    Calculate aggregate Greeks for a portfolio

    Args:
        positions: List of dicts with keys: S, K, T, r, sigma, type, quantity

    Returns:
        Portfolio-level Greeks
    """
    portfolio = {
        'delta': 0.0,
        'gamma': 0.0,
        'theta': 0.0,
        'vega': 0.0,
        'rho': 0.0
    }

    for pos in positions:
        greeks = OptionsAnalysis.calculate_greeks(
            S=pos['S'], K=pos['K'], T=pos['T'],
            r=pos['r'], sigma=pos['sigma'],
            option_type=pos['type']
        )

        # Multiply by position size (100 shares per contract)
        multiplier = pos['quantity'] * 100

        portfolio['delta'] += greeks['delta'] * multiplier
        portfolio['gamma'] += greeks['gamma'] * multiplier
        portfolio['theta'] += greeks['theta'] * multiplier
        portfolio['vega'] += greeks['vega'] * multiplier
        portfolio['rho'] += greeks['rho'] * multiplier

    return portfolio
```

## Best Practices

1. **Always validate inputs** before calculating Greeks (T > 0, sigma > 0, S > 0, K > 0)
2. **Handle edge cases** gracefully (T=0, extremely ITM/OTM options)
3. **Use vectorized operations** for calculating Greeks across multiple strikes/expirations
4. **Cache d1/d2 calculations** when computing multiple Greeks for same option
5. **Scale vega and rho** to per-1% changes for interpretability
6. **Convert theta to per-day** for practical usage (divide annual theta by 365)

## Common Pitfalls

❌ **Using annual theta** instead of daily theta
✅ Divide by 365 to get theta per calendar day

❌ **Forgetting to scale vega** to per-1% vol change
✅ Divide vega by 100 for industry-standard scaling

❌ **Not handling T=0 case** (expiration)
✅ Return intrinsic value when T <= 0

❌ **Using wrong sign conventions** for put Greeks
✅ Put delta is negative (-1 to 0), call delta is positive (0 to 1)

❌ **IV calculation on deep ITM/OTM** options (numerical instability)
✅ Add bounds checking and catch convergence failures

## Quality Standards

- **Accuracy**: <0.001 difference vs industry-standard libraries (QuantLib, Bloomberg)
- **IV Convergence**: 95% success rate on liquid options (bid-ask spread <10%)
- **Performance**: >1000 Greeks calculations per second
- **Edge Case Handling**: Graceful degradation for T=0, extreme moneyness
- **Type Safety**: 100% type-hinted with Literal for option_type

---

**Skill Type**: Finance - Options Pricing
**Complexity**: Complex
**Typical Usage**: Activated when quantitative-analyst needs options Greeks or implied volatility
**Performance**: Vectorized Black-Scholes with Newton-Raphson IV solver
