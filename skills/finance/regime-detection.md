---
name: regime-detection
description: Load when user needs market regime classification, regime-switching models, trend detection, or adaptive strategy selection. Covers regime identification and strategy adaptation.
trigger_keywords: [regime detection, market regime, regime switching, markov, trend detection, volatility regime, risk-on risk-off, bull market, bear market, sideways, mean reverting, trending, hurst exponent, adx, regime change, adaptive strategy]
---

# Regime Detection & Adaptive Strategies Skill

Market regime identification and strategy adaptation for robust performance across market conditions.

## Core Concepts

- **Regime Persistence**: Markets tend to stay in regimes (trending/ranging) longer than random; exploit this with adaptive strategies
- **Hurst Exponent**: H > 0.5 indicates trending (momentum works), H < 0.5 indicates mean-reverting, H = 0.5 is random walk
- **Volatility Clustering**: High volatility periods cluster together (GARCH effect); regime detection captures this
- **Strategy Selection**: Momentum strategies work in trends, mean-reversion works in ranges - wrong strategy loses money
- **Regime Transition**: The most dangerous period; existing positions may be wrong for the new regime

## Regime Framework

### Core Regime Types

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

class MarketRegime(Enum):
    """Primary market regime classification"""
    BULL_TREND = "bull_trend"           # Strong uptrend
    BEAR_TREND = "bear_trend"           # Strong downtrend
    HIGH_VOL_BULL = "high_vol_bull"     # Up with high volatility
    HIGH_VOL_BEAR = "high_vol_bear"     # Down with high volatility (crisis)
    LOW_VOL_RANGE = "low_vol_range"     # Sideways, low vol (mean reversion)
    HIGH_VOL_RANGE = "high_vol_range"   # Choppy, high vol

class TrendRegime(Enum):
    """Trend strength classification"""
    STRONG_TREND = "strong_trend"       # Momentum strategies work
    WEAK_TREND = "weak_trend"           # Mixed signals
    MEAN_REVERTING = "mean_reverting"   # Mean reversion works

@dataclass
class RegimeState:
    """Current regime assessment"""
    primary_regime: MarketRegime
    trend_regime: TrendRegime
    volatility_percentile: float  # 0-100
    correlation_regime: str       # 'high', 'normal', 'low'
    confidence: float             # 0-1
    timestamp: str
```

## Regime Detection Methods

### Volatility Regime (VIX-based)

```python
class VolatilityRegimeDetector:
    """Classify volatility regime using VIX or realized vol"""

    def __init__(
        self,
        low_threshold: float = 15,
        high_threshold: float = 25,
        extreme_threshold: float = 35
    ):
        self.low = low_threshold
        self.high = high_threshold
        self.extreme = extreme_threshold

    def classify(self, vix: float) -> str:
        """
        VIX Regime Classification:
        - < 15: Low vol (complacency, good for selling premium)
        - 15-25: Normal vol
        - 25-35: Elevated vol (caution)
        - > 35: Crisis vol (hedging expensive, opportunities)
        """
        if vix < self.low:
            return 'low_vol'
        elif vix < self.high:
            return 'normal_vol'
        elif vix < self.extreme:
            return 'elevated_vol'
        else:
            return 'crisis_vol'

    def classify_percentile(
        self,
        current_vol: float,
        historical_vol: pd.Series
    ) -> dict:
        """Classify based on historical percentile"""
        from scipy import stats
        percentile = stats.percentileofscore(historical_vol, current_vol)

        if percentile < 20:
            regime = 'very_low'
        elif percentile < 40:
            regime = 'low'
        elif percentile < 60:
            regime = 'normal'
        elif percentile < 80:
            regime = 'elevated'
        else:
            regime = 'extreme'

        return {
            'regime': regime,
            'percentile': percentile,
            'current_vol': current_vol,
            'historical_median': historical_vol.median()
        }
```

### Trend Regime (ADX/Hurst)

```python
def calculate_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> pd.Series:
    """
    Average Directional Index (ADX)

    ADX > 25: Trending market (momentum works)
    ADX < 20: Range-bound (mean reversion works)
    """
    # True Range
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    # Directional Movement
    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smoothed averages
    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx

def calculate_hurst_exponent(
    prices: pd.Series,
    max_lag: int = 100
) -> float:
    """
    Hurst Exponent for regime classification

    H < 0.5: Mean-reverting (mean reversion strategies)
    H = 0.5: Random walk (no edge)
    H > 0.5: Trending (momentum strategies)

    Uses R/S analysis
    """
    lags = range(2, max_lag)
    rs_values = []

    for lag in lags:
        # Divide into chunks
        n_chunks = len(prices) // lag
        if n_chunks < 2:
            continue

        rs_chunk = []
        for i in range(n_chunks):
            chunk = prices.iloc[i*lag:(i+1)*lag]
            returns = chunk.pct_change().dropna()

            if len(returns) < 2:
                continue

            # Cumulative deviation from mean
            mean_return = returns.mean()
            cumdev = (returns - mean_return).cumsum()

            # Range
            R = cumdev.max() - cumdev.min()

            # Standard deviation
            S = returns.std()

            if S > 0:
                rs_chunk.append(R / S)

        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))

    if len(rs_values) < 2:
        return 0.5  # Default to random walk

    # Log-log regression
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])

    from scipy.stats import linregress
    slope, _, _, _, _ = linregress(log_lags, log_rs)

    return slope

class TrendRegimeDetector:
    """Detect trending vs mean-reverting regime"""

    def __init__(self, adx_threshold: float = 25, hurst_threshold: float = 0.5):
        self.adx_threshold = adx_threshold
        self.hurst_threshold = hurst_threshold

    def classify(
        self,
        prices: pd.Series,
        high: pd.Series = None,
        low: pd.Series = None
    ) -> dict:
        """
        Combined ADX + Hurst classification
        """
        # Hurst exponent
        hurst = calculate_hurst_exponent(prices)

        # ADX if OHLC available
        if high is not None and low is not None:
            adx = calculate_adx(high, low, prices).iloc[-1]
        else:
            adx = None

        # Classification
        if hurst > 0.55 and (adx is None or adx > self.adx_threshold):
            regime = TrendRegime.STRONG_TREND
            strategy_rec = 'momentum'
        elif hurst < 0.45:
            regime = TrendRegime.MEAN_REVERTING
            strategy_rec = 'mean_reversion'
        else:
            regime = TrendRegime.WEAK_TREND
            strategy_rec = 'reduce_exposure'

        return {
            'regime': regime,
            'hurst': hurst,
            'adx': adx,
            'strategy_recommendation': strategy_rec
        }
```

### Markov Regime Switching

```python
class MarkovRegimeSwitching:
    """
    Hidden Markov Model for regime detection

    Assumes market switches between N hidden states
    Each state has different return/volatility characteristics
    """

    def __init__(self, n_regimes: int = 2):
        self.n_regimes = n_regimes
        self.model = None

    def fit(self, returns: pd.Series):
        """Fit HMM to return series"""
        from hmmlearn import hmm

        # Reshape for sklearn
        X = returns.values.reshape(-1, 1)

        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100
        )
        self.model.fit(X)

        # Get regime characteristics
        self.regime_means = self.model.means_.flatten()
        self.regime_vols = np.sqrt(self.model.covars_.flatten())

        # Label regimes (0 = low vol, 1 = high vol typically)
        vol_order = np.argsort(self.regime_vols)
        self.regime_labels = {
            vol_order[0]: 'low_vol_regime',
            vol_order[1]: 'high_vol_regime'
        }

        return self

    def predict_regime(self, returns: pd.Series) -> pd.Series:
        """Predict regime for each observation"""
        X = returns.values.reshape(-1, 1)
        regimes = self.model.predict(X)
        return pd.Series(regimes, index=returns.index)

    def regime_probabilities(self, returns: pd.Series) -> pd.DataFrame:
        """Get probability of each regime"""
        X = returns.values.reshape(-1, 1)
        probs = self.model.predict_proba(X)
        return pd.DataFrame(
            probs,
            index=returns.index,
            columns=[f'regime_{i}_prob' for i in range(self.n_regimes)]
        )

    def current_regime(self, returns: pd.Series) -> dict:
        """Get current regime and characteristics"""
        regime_idx = self.predict_regime(returns).iloc[-1]
        probs = self.regime_probabilities(returns).iloc[-1]

        return {
            'regime': self.regime_labels.get(regime_idx, f'regime_{regime_idx}'),
            'regime_idx': regime_idx,
            'confidence': probs.max(),
            'regime_mean': self.regime_means[regime_idx] * 252,  # Annualized
            'regime_vol': self.regime_vols[regime_idx] * np.sqrt(252),
            'transition_prob': self.model.transmat_[regime_idx]
        }
```

## Composite Regime Classifier

```python
class CompositeRegimeClassifier:
    """Combine multiple regime signals"""

    def __init__(self):
        self.vol_detector = VolatilityRegimeDetector()
        self.trend_detector = TrendRegimeDetector()
        self.hmm_detector = MarkovRegimeSwitching(n_regimes=2)

    def classify(
        self,
        prices: pd.Series,
        returns: pd.Series,
        vix: float = None,
        high: pd.Series = None,
        low: pd.Series = None
    ) -> RegimeState:
        """
        Comprehensive regime classification
        """
        # Volatility regime
        if vix is not None:
            vol_regime = self.vol_detector.classify(vix)
            vol_percentile = self.vol_detector.classify_percentile(
                vix,
                pd.Series([15, 20, 25, 30, 35])  # Placeholder
            )['percentile']
        else:
            realized_vol = returns.rolling(21).std().iloc[-1] * np.sqrt(252) * 100
            vol_regime = self.vol_detector.classify(realized_vol)
            vol_percentile = 50  # Default

        # Trend regime
        trend_result = self.trend_detector.classify(prices, high, low)

        # HMM regime
        if len(returns) > 100:
            self.hmm_detector.fit(returns)
            hmm_result = self.hmm_detector.current_regime(returns)
        else:
            hmm_result = {'regime': 'unknown', 'confidence': 0}

        # Combine into primary regime
        primary_regime = self._combine_signals(
            vol_regime,
            trend_result['regime'],
            returns.iloc[-21:].sum()  # Recent return direction
        )

        return RegimeState(
            primary_regime=primary_regime,
            trend_regime=trend_result['regime'],
            volatility_percentile=vol_percentile,
            correlation_regime='normal',  # Would need correlation data
            confidence=hmm_result.get('confidence', 0.5),
            timestamp=str(prices.index[-1])
        )

    def _combine_signals(
        self,
        vol_regime: str,
        trend_regime: TrendRegime,
        recent_return: float
    ) -> MarketRegime:
        """Combine signals into single regime"""

        is_high_vol = vol_regime in ['elevated_vol', 'crisis_vol']
        is_trending = trend_regime == TrendRegime.STRONG_TREND
        is_up = recent_return > 0

        if is_high_vol and is_up:
            return MarketRegime.HIGH_VOL_BULL
        elif is_high_vol and not is_up:
            return MarketRegime.HIGH_VOL_BEAR
        elif not is_high_vol and is_trending and is_up:
            return MarketRegime.BULL_TREND
        elif not is_high_vol and is_trending and not is_up:
            return MarketRegime.BEAR_TREND
        elif is_high_vol:
            return MarketRegime.HIGH_VOL_RANGE
        else:
            return MarketRegime.LOW_VOL_RANGE
```

## Adaptive Strategy Selection

```python
class AdaptiveStrategySelector:
    """Select strategy parameters based on regime"""

    def __init__(self):
        # Strategy recommendations by regime
        self.regime_strategies = {
            MarketRegime.BULL_TREND: {
                'primary': 'momentum',
                'position_size': 1.0,
                'stop_loss_multiplier': 1.5,  # Wider stops in trend
                'take_profit': None  # Let winners run
            },
            MarketRegime.BEAR_TREND: {
                'primary': 'momentum_short',
                'position_size': 0.5,  # Reduced size
                'stop_loss_multiplier': 1.0,
                'take_profit': 2.0  # Take profits faster
            },
            MarketRegime.HIGH_VOL_BULL: {
                'primary': 'reduced_momentum',
                'position_size': 0.5,
                'stop_loss_multiplier': 2.0,  # Very wide for vol
                'take_profit': 1.5
            },
            MarketRegime.HIGH_VOL_BEAR: {
                'primary': 'defensive',
                'position_size': 0.25,
                'stop_loss_multiplier': 2.5,
                'take_profit': 1.0  # Quick exits
            },
            MarketRegime.LOW_VOL_RANGE: {
                'primary': 'mean_reversion',
                'position_size': 1.0,
                'stop_loss_multiplier': 1.0,
                'take_profit': 1.0
            },
            MarketRegime.HIGH_VOL_RANGE: {
                'primary': 'reduced_all',
                'position_size': 0.3,
                'stop_loss_multiplier': 2.0,
                'take_profit': 0.75
            }
        }

    def get_strategy_params(self, regime: RegimeState) -> dict:
        """Get strategy parameters for current regime"""
        base_params = self.regime_strategies.get(
            regime.primary_regime,
            self.regime_strategies[MarketRegime.LOW_VOL_RANGE]
        )

        # Adjust for confidence
        if regime.confidence < 0.6:
            base_params['position_size'] *= 0.7  # Reduce if uncertain

        return base_params

    def blend_strategies(
        self,
        momentum_signal: float,
        mean_reversion_signal: float,
        regime: RegimeState
    ) -> float:
        """
        Blend momentum and mean reversion based on regime

        Returns combined signal (-1 to 1)
        """
        if regime.trend_regime == TrendRegime.STRONG_TREND:
            weights = {'momentum': 0.8, 'mean_reversion': 0.2}
        elif regime.trend_regime == TrendRegime.MEAN_REVERTING:
            weights = {'momentum': 0.2, 'mean_reversion': 0.8}
        else:
            weights = {'momentum': 0.5, 'mean_reversion': 0.5}

        return (
            weights['momentum'] * momentum_signal +
            weights['mean_reversion'] * mean_reversion_signal
        )
```

## Production Implementation

```python
class RegimeAwareTrading:
    """Production regime-aware trading system"""

    def __init__(self):
        self.classifier = CompositeRegimeClassifier()
        self.selector = AdaptiveStrategySelector()
        self.current_regime = None
        self.regime_history = []

    def update(
        self,
        prices: pd.Series,
        returns: pd.Series,
        vix: float = None
    ) -> dict:
        """Update regime and get trading parameters"""

        # Classify current regime
        self.current_regime = self.classifier.classify(
            prices, returns, vix
        )

        # Track regime changes
        self.regime_history.append({
            'timestamp': self.current_regime.timestamp,
            'regime': self.current_regime.primary_regime.value,
            'confidence': self.current_regime.confidence
        })

        # Get strategy parameters
        params = self.selector.get_strategy_params(self.current_regime)

        return {
            'regime': self.current_regime,
            'strategy_params': params,
            'regime_changed': self._detect_regime_change()
        }

    def _detect_regime_change(self) -> bool:
        """Detect if regime recently changed"""
        if len(self.regime_history) < 2:
            return False

        return (
            self.regime_history[-1]['regime'] !=
            self.regime_history[-2]['regime']
        )
```

## Best Practices

1. **Multiple signals**: Combine vol, trend, and statistical regime indicators
2. **Confidence weighting**: Reduce size when regime is uncertain
3. **Lag awareness**: Regime detection has lag; don't over-trade on changes
4. **Backtest per regime**: Validate strategies work in each regime separately
5. **Smooth transitions**: Don't flip strategies instantly; blend gradually

## Common Pitfalls

- **Overfitting regimes**: Too many regime states = noise fitting
- **Hindsight bias**: Regimes obvious in retrospect, hard to detect live
- **Whipsaw**: Regime flipping causes excessive trading
- **Single indicator**: One signal (e.g., VIX only) is insufficient
- **Ignoring transitions**: Strategy needs time to adapt to new regime

---

**Skill Type**: Finance - Regime Detection
**Complexity**: Advanced
**Typical Usage**: Adaptive strategy selection, risk management
