---
name: ml-trading
description: Load when user needs machine learning for trading, feature engineering, time-series ML, or trading-specific model validation. Covers ML pipelines for alpha generation.
trigger_keywords: [machine learning trading, ml trading, feature engineering, time series ml, xgboost trading, neural network trading, lstm trading, random forest trading, walk forward validation, purged cross validation, alpha model, signal generation ml, trading features]
---

# Machine Learning for Trading Skill

ML pipelines for trading with proper time-series validation, feature engineering, and production deployment.

## Time-Series Cross-Validation

### Purged K-Fold (Critical for Trading)

```python
import numpy as np
import pandas as pd
from typing import List, Tuple, Generator
from sklearn.model_selection import BaseCrossValidator

class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation for Time Series

    CRITICAL: Standard K-Fold leaks future information!

    This implementation:
    1. Respects time ordering (train always before test)
    2. Purges overlapping labels (removes contamination)
    3. Adds embargo period (gap between train/test)
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 5
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test indices with purging and embargo
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Test fold size
        test_size = n_samples // self.n_splits

        for i in range(self.n_splits):
            # Test indices
            test_start = i * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n_samples
            test_indices = indices[test_start:test_end]

            # Train indices (before test, with purge and embargo)
            train_end = test_start - self.purge_days - self.embargo_days
            if train_end <= 0:
                continue  # Skip if not enough training data

            train_indices = indices[:train_end]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


class WalkForwardCV(BaseCrossValidator):
    """
    Walk-Forward Validation (Expanding or Rolling Window)

    Most realistic for trading: train on past, test on future
    """

    def __init__(
        self,
        n_splits: int = 10,
        train_size: int = None,  # None = expanding window
        test_size: int = 21,     # ~1 month
        gap: int = 5             # Embargo gap
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.test_size = test_size
        self.gap = gap

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series = None,
        groups = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Walk-forward split generator"""
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Calculate minimum training size
        min_train = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Test window
            test_end = n_samples - i * self.test_size
            test_start = test_end - self.test_size

            if test_start <= min_train:
                continue

            # Train window
            train_end = test_start - self.gap

            if self.train_size is not None:
                # Rolling window
                train_start = max(0, train_end - self.train_size)
            else:
                # Expanding window
                train_start = 0

            train_indices = indices[train_start:train_end]
            test_indices = indices[test_start:test_end]

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits
```

## Feature Engineering

### Price-Based Features

```python
class TradingFeatureEngine:
    """Feature engineering for trading ML models"""

    def __init__(self, prices: pd.DataFrame):
        """
        prices: DataFrame with columns [open, high, low, close, volume]
        """
        self.prices = prices
        self.features = pd.DataFrame(index=prices.index)

    def add_returns(self, periods: List[int] = [1, 5, 10, 21, 63]) -> 'TradingFeatureEngine':
        """
        Return features at multiple horizons

        IMPORTANT: Use log returns for stationarity
        """
        close = self.prices['close']

        for p in periods:
            self.features[f'ret_{p}d'] = np.log(close / close.shift(p))

        return self

    def add_volatility(self, windows: List[int] = [5, 10, 21, 63]) -> 'TradingFeatureEngine':
        """Realized volatility features"""
        returns = np.log(self.prices['close'] / self.prices['close'].shift(1))

        for w in windows:
            self.features[f'vol_{w}d'] = returns.rolling(w).std() * np.sqrt(252)

        # Volatility ratio (short/long)
        self.features['vol_ratio'] = self.features['vol_5d'] / self.features['vol_21d']

        return self

    def add_momentum(self) -> 'TradingFeatureEngine':
        """Momentum indicators"""
        close = self.prices['close']

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        self.features['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        self.features['macd'] = ema_12 - ema_26
        self.features['macd_signal'] = self.features['macd'].ewm(span=9).mean()
        self.features['macd_hist'] = self.features['macd'] - self.features['macd_signal']

        # Rate of Change
        self.features['roc_10'] = (close - close.shift(10)) / close.shift(10)
        self.features['roc_21'] = (close - close.shift(21)) / close.shift(21)

        return self

    def add_mean_reversion(self) -> 'TradingFeatureEngine':
        """Mean reversion features"""
        close = self.prices['close']

        # Bollinger Band position
        sma_20 = close.rolling(20).mean()
        std_20 = close.rolling(20).std()
        self.features['bb_position'] = (close - sma_20) / (2 * std_20)

        # Distance from moving averages
        for ma in [10, 20, 50, 200]:
            sma = close.rolling(ma).mean()
            self.features[f'dist_sma_{ma}'] = (close - sma) / sma

        # Z-score of returns
        ret_21 = np.log(close / close.shift(21))
        self.features['ret_zscore'] = (ret_21 - ret_21.rolling(252).mean()) / ret_21.rolling(252).std()

        return self

    def add_volume_features(self) -> 'TradingFeatureEngine':
        """Volume-based features"""
        volume = self.prices['volume']
        close = self.prices['close']

        # Volume ratio
        self.features['vol_ratio_10'] = volume / volume.rolling(10).mean()

        # On-Balance Volume trend
        obv = (np.sign(close.diff()) * volume).cumsum()
        self.features['obv_slope'] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )

        # Dollar volume
        dollar_vol = close * volume
        self.features['dollar_vol_ratio'] = dollar_vol / dollar_vol.rolling(21).mean()

        return self

    def add_microstructure(self) -> 'TradingFeatureEngine':
        """Market microstructure features"""
        high = self.prices['high']
        low = self.prices['low']
        close = self.prices['close']

        # True Range / ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        self.features['atr_14'] = tr.rolling(14).mean()

        # Intraday volatility
        self.features['intraday_range'] = (high - low) / close

        # Gap (overnight)
        self.features['gap'] = self.prices['open'] / close.shift(1) - 1

        return self

    def add_all_features(self) -> pd.DataFrame:
        """Add all features and return"""
        return (self
                .add_returns()
                .add_volatility()
                .add_momentum()
                .add_mean_reversion()
                .add_volume_features()
                .add_microstructure()
                .features)
```

### Target Engineering

```python
def create_target(
    prices: pd.Series,
    horizon: int = 5,
    target_type: str = 'return'
) -> pd.Series:
    """
    Create prediction target

    CRITICAL: Shift target to avoid look-ahead bias
    Target should be FUTURE return, not past
    """
    if target_type == 'return':
        # Forward return
        target = prices.shift(-horizon) / prices - 1
    elif target_type == 'direction':
        # Binary direction
        target = (prices.shift(-horizon) > prices).astype(int)
    elif target_type == 'tercile':
        # Tercile classification (down/flat/up)
        returns = prices.shift(-horizon) / prices - 1
        target = pd.cut(returns, bins=3, labels=[0, 1, 2]).astype(float)
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    return target

def create_labels_with_barriers(
    prices: pd.Series,
    horizon: int = 21,
    profit_target: float = 0.02,
    stop_loss: float = 0.02
) -> pd.Series:
    """
    Triple-barrier labeling (from Advances in Financial ML)

    Labels based on which barrier is hit first:
    - Upper barrier (profit target): +1
    - Lower barrier (stop loss): -1
    - Time barrier (horizon): 0
    """
    labels = pd.Series(index=prices.index, dtype=float)

    for i in range(len(prices) - horizon):
        entry_price = prices.iloc[i]
        future_prices = prices.iloc[i+1:i+horizon+1]

        # Check barriers
        returns = future_prices / entry_price - 1

        # Upper barrier hit?
        upper_hits = returns >= profit_target
        if upper_hits.any():
            upper_first = upper_hits.idxmax()
        else:
            upper_first = None

        # Lower barrier hit?
        lower_hits = returns <= -stop_loss
        if lower_hits.any():
            lower_first = lower_hits.idxmax()
        else:
            lower_first = None

        # Determine label
        if upper_first is not None and lower_first is not None:
            # Both hit - which first?
            if upper_first < lower_first:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = -1
        elif upper_first is not None:
            labels.iloc[i] = 1
        elif lower_first is not None:
            labels.iloc[i] = -1
        else:
            labels.iloc[i] = 0  # Time barrier

    return labels
```

## ML Models for Trading

### XGBoost Classifier

```python
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score

class TradingXGBoost:
    """XGBoost for trading signals"""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8
    ):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.feature_importance = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None
    ):
        """Train with early stopping if validation set provided"""
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=10 if X_val is not None else None,
            verbose=False
        )

        # Store feature importance
        self.feature_importance = pd.Series(
            self.model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions"""
        return self.model.predict_proba(X)

    def generate_signal(
        self,
        X: pd.DataFrame,
        threshold: float = 0.6
    ) -> pd.Series:
        """
        Generate trading signal from predictions

        Signal only when confident (prob > threshold)
        """
        proba = self.predict_proba(X)

        signals = pd.Series(index=X.index, data=0)
        signals[proba[:, 1] > threshold] = 1   # Long signal
        signals[proba[:, 0] > threshold] = -1  # Short signal

        return signals
```

### LSTM for Sequences

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class TradingLSTM:
    """LSTM for sequence prediction"""

    def __init__(
        self,
        sequence_length: int = 20,
        n_features: int = 10,
        lstm_units: int = 50
    ):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = self._build_model(lstm_units)

    def _build_model(self, lstm_units: int) -> Sequential:
        """Build LSTM architecture"""
        model = Sequential([
            LSTM(lstm_units, return_sequences=True,
                 input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(3, activation='softmax')  # Down/Flat/Up
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def prepare_sequences(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convert DataFrame to sequences for LSTM"""
        X, y = [], []

        for i in range(self.sequence_length, len(features)):
            X.append(features.iloc[i-self.sequence_length:i].values)
            y.append(target.iloc[i])

        return np.array(X), np.array(y)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2
    ):
        """Train LSTM"""
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        return self
```

## Model Evaluation

```python
class TradingModelEvaluator:
    """Evaluate ML trading models"""

    def __init__(self, predictions: pd.Series, actuals: pd.Series, returns: pd.Series):
        self.predictions = predictions
        self.actuals = actuals
        self.returns = returns

    def classification_metrics(self) -> dict:
        """Standard classification metrics"""
        return {
            'accuracy': accuracy_score(self.actuals, self.predictions),
            'precision': precision_score(self.actuals, self.predictions, average='weighted'),
            'recall': recall_score(self.actuals, self.predictions, average='weighted')
        }

    def trading_metrics(self) -> dict:
        """Trading-specific metrics"""
        # Strategy returns
        strategy_returns = self.predictions.shift(1) * self.returns
        cumulative = (1 + strategy_returns).cumprod()

        # Sharpe ratio
        sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

        # Max drawdown
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Win rate
        wins = (strategy_returns > 0).sum()
        total_trades = (self.predictions != 0).sum()
        win_rate = wins / total_trades if total_trades > 0 else 0

        return {
            'total_return': cumulative.iloc[-1] - 1,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'n_trades': total_trades
        }

    def information_coefficient(self) -> float:
        """
        Information Coefficient (IC)

        Correlation between predictions and actual returns
        IC > 0.05 is meaningful for trading
        """
        return self.predictions.corr(self.returns.shift(-1))
```

## Production Pipeline

```python
class MLTradingPipeline:
    """End-to-end ML trading pipeline"""

    def __init__(
        self,
        model_type: str = 'xgboost',
        retrain_frequency: str = 'monthly'
    ):
        self.model_type = model_type
        self.retrain_frequency = retrain_frequency
        self.model = None
        self.feature_engine = None

    def fit(self, prices: pd.DataFrame) -> 'MLTradingPipeline':
        """Fit pipeline on historical data"""

        # Feature engineering
        self.feature_engine = TradingFeatureEngine(prices)
        features = self.feature_engine.add_all_features()

        # Create target (5-day forward return direction)
        target = create_target(prices['close'], horizon=5, target_type='direction')

        # Align and clean
        data = pd.concat([features, target.rename('target')], axis=1).dropna()

        X = data.drop('target', axis=1)
        y = data['target']

        # Walk-forward validation
        cv = WalkForwardCV(n_splits=5, test_size=63)

        cv_scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if self.model_type == 'xgboost':
                model = TradingXGBoost()
                model.train(X_train, y_train)
                preds = model.model.predict(X_test)
                cv_scores.append(accuracy_score(y_test, preds))

        print(f"CV Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")

        # Final model on all data
        if self.model_type == 'xgboost':
            self.model = TradingXGBoost()
            self.model.train(X, y)

        return self

    def predict(self, prices: pd.DataFrame) -> pd.Series:
        """Generate predictions for new data"""
        features = TradingFeatureEngine(prices).add_all_features()
        return self.model.generate_signal(features.dropna())
```

## Best Practices

1. **Purged CV is mandatory**: Standard K-Fold leaks future info
2. **Feature lag**: Ensure all features use only past data
3. **Simple models first**: XGBoost often beats deep learning
4. **IC > 0.03**: Minimum information coefficient for tradeable signal
5. **Out-of-sample only**: Never judge model by in-sample performance

## Common Pitfalls

- **Look-ahead bias**: Using future data in features (most common error!)
- **Overfitting**: Too many features, too little data
- **Ignoring costs**: Model profitable before costs, negative after
- **Data snooping**: Testing many models, reporting best
- **Non-stationary features**: Raw prices as features (use returns!)

---

**Skill Type**: Finance - Machine Learning
**Complexity**: Advanced
**Typical Usage**: Alpha generation, signal enhancement, pattern recognition
