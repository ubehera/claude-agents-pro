---
name: volatility-modeling
description: Load when user needs GARCH models, volatility surfaces, implied vol, term structure, or smile/skew dynamics. Covers volatility forecasting and surface construction.
trigger_keywords: [garch, egarch, volatility surface, vol surface, implied volatility, iv surface, volatility smile, volatility skew, term structure, realized volatility, historical volatility, vix, volatility forecasting, stochastic volatility, heston, local volatility]
---

# Volatility Modeling Skill

Advanced volatility modeling with GARCH, volatility surfaces, and term structure analysis.

## Core Concepts

- **Implied vs Realized**: Implied volatility (from options prices) reflects expected future vol; realized is actual historical vol. IV typically exceeds RV (volatility risk premium)
- **Volatility Clustering**: High volatility tends to follow high volatility (GARCH effect); use for vol forecasting
- **Term Structure**: Short-term IV vs long-term IV; steep term structure suggests mean reversion, flat/inverted suggests stress
- **Volatility Smile/Skew**: OTM puts typically have higher IV than OTM calls (crash protection premium); quantify with risk reversal
- **Yang-Zhang Estimator**: Best volatility estimator for stocks with overnight gaps; uses OHLC data efficiently

## Realized Volatility

### Historical Volatility Estimators

```python
import numpy as np
import pandas as pd
from typing import Literal

def close_to_close_vol(
    prices: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Standard close-to-close volatility

    σ = std(log returns) * √252
    """
    log_returns = np.log(prices / prices.shift(1))
    vol = log_returns.rolling(window).std()

    if annualize:
        vol *= np.sqrt(252)

    return vol

def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Parkinson volatility estimator (uses high-low range)

    More efficient than close-to-close (uses more information)
    σ² = (1/4ln2) * E[(ln(H/L))²]
    """
    log_hl = np.log(high / low)
    variance = (log_hl ** 2) / (4 * np.log(2))
    vol = np.sqrt(variance.rolling(window).mean())

    if annualize:
        vol *= np.sqrt(252)

    return vol

def garman_klass_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Garman-Klass volatility estimator

    Uses OHLC data - most efficient estimator for continuous prices
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)

    variance = 0.5 * log_hl**2 - (2*np.log(2) - 1) * log_co**2
    vol = np.sqrt(variance.rolling(window).mean())

    if annualize:
        vol *= np.sqrt(252)

    return vol

def yang_zhang_vol(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 21,
    annualize: bool = True
) -> pd.Series:
    """
    Yang-Zhang volatility estimator

    Handles overnight jumps - best for stocks with gaps
    """
    log_oc = np.log(open_ / close.shift(1))  # Overnight
    log_co = np.log(close / open_)            # Intraday close-open
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)

    # Overnight variance
    overnight_var = log_oc.rolling(window).var()

    # Open-to-close variance
    open_close_var = log_co.rolling(window).var()

    # Rogers-Satchell variance
    rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variance = overnight_var + k * open_close_var + (1 - k) * rs_var

    vol = np.sqrt(variance)

    if annualize:
        vol *= np.sqrt(252)

    return vol
```

## GARCH Models

### GARCH(1,1)

```python
from arch import arch_model

class GARCHModel:
    """GARCH volatility forecasting"""

    def __init__(self, returns: pd.Series):
        """
        returns: Daily log returns (not annualized)
        """
        self.returns = returns * 100  # Scale for numerical stability
        self.model = None
        self.result = None

    def fit_garch(
        self,
        p: int = 1,
        q: int = 1,
        dist: str = 'normal'  # 'normal', 't', 'skewt'
    ):
        """
        Fit GARCH(p,q) model

        σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

        Where:
        - ω (omega): Long-run variance weight
        - α (alpha): Shock impact (news coefficient)
        - β (beta): Persistence
        - α + β < 1 for stationarity
        """
        self.model = arch_model(
            self.returns,
            vol='Garch',
            p=p,
            q=q,
            dist=dist
        )
        self.result = self.model.fit(disp='off')

        return {
            'omega': self.result.params['omega'],
            'alpha': self.result.params['alpha[1]'],
            'beta': self.result.params['beta[1]'],
            'persistence': self.result.params['alpha[1]'] + self.result.params['beta[1]'],
            'long_run_vol': np.sqrt(
                self.result.params['omega'] /
                (1 - self.result.params['alpha[1]'] - self.result.params['beta[1]'])
            ) * np.sqrt(252) / 100,
            'aic': self.result.aic,
            'bic': self.result.bic
        }

    def fit_egarch(self, p: int = 1, q: int = 1):
        """
        EGARCH - Exponential GARCH

        Captures asymmetric volatility (leverage effect)
        Bad news increases vol more than good news
        """
        self.model = arch_model(
            self.returns,
            vol='EGARCH',
            p=p,
            q=q
        )
        self.result = self.model.fit(disp='off')

        return {
            'omega': self.result.params['omega'],
            'alpha': self.result.params['alpha[1]'],
            'gamma': self.result.params['gamma[1]'],  # Asymmetry
            'beta': self.result.params['beta[1]']
        }

    def fit_gjr_garch(self, p: int = 1, q: int = 1):
        """
        GJR-GARCH (Threshold GARCH)

        σ²_t = ω + (α + γ*I_{t-1}) * ε²_{t-1} + β * σ²_{t-1}

        Where I_{t-1} = 1 if ε_{t-1} < 0 (bad news indicator)
        """
        self.model = arch_model(
            self.returns,
            vol='Garch',
            p=p,
            o=1,  # One asymmetric term
            q=q
        )
        self.result = self.model.fit(disp='off')

        return self.result.params

    def forecast(self, horizon: int = 5) -> pd.DataFrame:
        """
        Forecast volatility h periods ahead

        Returns annualized volatility forecasts
        """
        if self.result is None:
            raise ValueError("Must fit model first")

        forecasts = self.result.forecast(horizon=horizon)

        # Convert variance to annualized vol
        vol_forecasts = np.sqrt(forecasts.variance.iloc[-1]) * np.sqrt(252) / 100

        return vol_forecasts

    def conditional_volatility(self) -> pd.Series:
        """Get fitted conditional volatility series"""
        if self.result is None:
            raise ValueError("Must fit model first")

        return np.sqrt(self.result.conditional_volatility) * np.sqrt(252) / 100
```

## Volatility Surface

### Surface Construction

```python
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.optimize import brentq

class VolatilitySurface:
    """Implied volatility surface from option prices"""

    def __init__(self):
        self.surface_data = None
        self.interpolator = None

    def build_surface(
        self,
        strikes: np.ndarray,
        expiries: np.ndarray,  # In years
        ivs: np.ndarray,
        spot: float
    ):
        """
        Build IV surface from market data

        Args:
            strikes: Array of strike prices
            expiries: Array of times to expiry (years)
            ivs: Array of implied volatilities
            spot: Current spot price
        """
        # Convert to moneyness
        moneyness = np.log(strikes / spot)

        self.surface_data = pd.DataFrame({
            'strike': strikes,
            'expiry': expiries,
            'moneyness': moneyness,
            'iv': ivs
        })

        # Build interpolator
        # Grid interpolation for irregular data
        self.interpolator = {
            'moneyness': moneyness,
            'expiry': expiries,
            'iv': ivs
        }

    def get_iv(
        self,
        strike: float,
        expiry: float,
        spot: float
    ) -> float:
        """
        Get interpolated IV for given strike and expiry
        """
        moneyness = np.log(strike / spot)

        # Use griddata for interpolation
        iv = griddata(
            points=(self.interpolator['moneyness'], self.interpolator['expiry']),
            values=self.interpolator['iv'],
            xi=(moneyness, expiry),
            method='cubic'
        )

        return float(iv)

    def term_structure(self, spot: float, moneyness: float = 0) -> pd.DataFrame:
        """
        Extract ATM term structure (IV vs expiry)
        """
        atm_data = self.surface_data[
            np.abs(self.surface_data['moneyness'] - moneyness) < 0.05
        ].sort_values('expiry')

        return atm_data[['expiry', 'iv']]

    def smile_at_expiry(self, expiry: float) -> pd.DataFrame:
        """
        Extract volatility smile at specific expiry
        """
        expiry_data = self.surface_data[
            np.abs(self.surface_data['expiry'] - expiry) < 0.01
        ].sort_values('moneyness')

        return expiry_data[['moneyness', 'strike', 'iv']]
```

### Smile/Skew Analysis

```python
class SmileAnalyzer:
    """Analyze volatility smile characteristics"""

    def __init__(self, surface: VolatilitySurface):
        self.surface = surface

    def calculate_skew(self, expiry: float, spot: float) -> dict:
        """
        Calculate volatility skew metrics

        Skew = IV(25Δ Put) - IV(25Δ Call)
        """
        smile = self.surface.smile_at_expiry(expiry)

        # Approximate 25-delta points (roughly 5% OTM)
        atm_strike = spot
        otm_put_strike = spot * 0.95
        otm_call_strike = spot * 1.05

        atm_iv = self.surface.get_iv(atm_strike, expiry, spot)
        put_iv = self.surface.get_iv(otm_put_strike, expiry, spot)
        call_iv = self.surface.get_iv(otm_call_strike, expiry, spot)

        return {
            'atm_iv': atm_iv,
            'put_iv': put_iv,
            'call_iv': call_iv,
            'skew': put_iv - call_iv,  # Positive = put skew
            'smile': (put_iv + call_iv) / 2 - atm_iv,  # Positive = smile (not smirk)
            'risk_reversal': call_iv - put_iv  # Call - Put
        }

    def term_structure_slope(self, spot: float) -> float:
        """
        Calculate slope of ATM term structure

        Positive = contango (far > near)
        Negative = backwardation (near > far)
        """
        ts = self.surface.term_structure(spot)

        if len(ts) < 2:
            return 0

        # Linear regression of IV vs expiry
        from scipy.stats import linregress
        slope, _, _, _, _ = linregress(ts['expiry'], ts['iv'])

        return slope
```

## Forward Volatility

```python
def forward_variance(
    iv_near: float,
    t_near: float,
    iv_far: float,
    t_far: float
) -> float:
    """
    Calculate forward variance between two expiries

    Forward Vol² = (σ²_far * T_far - σ²_near * T_near) / (T_far - T_near)

    Used for calendar spread pricing
    """
    var_near = iv_near ** 2 * t_near
    var_far = iv_far ** 2 * t_far

    forward_var = (var_far - var_near) / (t_far - t_near)

    return np.sqrt(forward_var) if forward_var > 0 else 0

def variance_swap_rate(
    iv_surface: VolatilitySurface,
    expiry: float,
    spot: float,
    num_strikes: int = 50
) -> float:
    """
    Calculate fair variance swap rate from options

    Integral over all strikes weighted by 1/K²
    """
    # Generate strike range
    strikes = np.linspace(spot * 0.5, spot * 1.5, num_strikes)

    variance_sum = 0
    for i in range(len(strikes) - 1):
        k = (strikes[i] + strikes[i+1]) / 2
        dk = strikes[i+1] - strikes[i]
        iv = iv_surface.get_iv(k, expiry, spot)

        # Contribution weighted by 1/K²
        variance_sum += (iv ** 2) * dk / (k ** 2)

    # Scale by 2/T
    fair_var = (2 / expiry) * spot * variance_sum

    return np.sqrt(fair_var)
```

## Volatility Forecasting

```python
class VolatilityForecaster:
    """Combine multiple volatility estimates"""

    def __init__(self):
        self.models = {}

    def ensemble_forecast(
        self,
        returns: pd.Series,
        implied_vol: float,
        horizon_days: int = 21
    ) -> dict:
        """
        Combine multiple vol forecasts

        Research shows optimal blend of:
        - GARCH (captures clustering)
        - Realized vol (recent history)
        - Implied vol (market expectation)
        """
        # GARCH forecast
        garch = GARCHModel(returns)
        garch.fit_garch()
        garch_forecast = garch.forecast(horizon_days).mean()

        # Realized vol (20-day)
        realized_vol = close_to_close_vol(
            np.exp(returns.cumsum()),  # Convert to prices
            window=20
        ).iloc[-1]

        # Optimal weights (from research)
        # IV typically gets highest weight
        weights = {
            'garch': 0.25,
            'realized': 0.25,
            'implied': 0.50
        }

        ensemble = (
            weights['garch'] * garch_forecast +
            weights['realized'] * realized_vol +
            weights['implied'] * implied_vol
        )

        return {
            'garch': garch_forecast,
            'realized': realized_vol,
            'implied': implied_vol,
            'ensemble': ensemble,
            'weights': weights
        }

    def vol_of_vol(self, vol_series: pd.Series, window: int = 21) -> pd.Series:
        """
        Volatility of volatility (vol-of-vol)

        High vol-of-vol = unstable regime
        Used for sizing vega positions
        """
        return vol_series.rolling(window).std()
```

## Best Practices

1. **Use multiple estimators**: Yang-Zhang for stocks with gaps, Parkinson for continuous
2. **GARCH for forecasting**: Better than simple historical vol
3. **Respect the smile**: Don't use flat vol for OTM options
4. **Forward vol for calendars**: Critical for calendar spread pricing
5. **Vol-of-vol for sizing**: Reduce vega exposure when vol-of-vol is high

## Common Pitfalls

- **Flat vol assumption**: Real markets have smile/skew
- **Ignoring term structure**: Near-term vs long-term vol differ significantly
- **Stale IV data**: IV changes rapidly; use real-time if trading
- **GARCH overfitting**: Simple GARCH(1,1) often beats complex variants
- **Ignoring jumps**: GARCH doesn't capture discontinuous moves

---

**Skill Type**: Finance - Volatility Modeling
**Complexity**: Advanced
**Typical Usage**: Options pricing, risk management, vol trading
