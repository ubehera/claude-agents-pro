# Finance Core Library

Shared core library for finance trading agents. Eliminates code duplication across 8+ agents by providing standardized schemas, performance metrics, and risk calculations.

## Overview

**Purpose**: Provide type-safe, production-ready utilities for quantitative trading analysis.

**Key Features**:
- Canonical data schemas (market data, signals, positions, options)
- Risk-adjusted performance metrics (Sharpe, Sortino, Calmar)
- Risk measurement tools (VaR, CVaR, beta)
- Trade statistics (win rate, profit factor, expectancy)

## Installation

```bash
# Navigate to core directory
cd /Users/umank/Code/agent-repos/ubehera/examples/finance/core

# Install in editable mode (recommended for development)
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

## Usage Examples

### Schemas

```python
from finance.core import MarketData, TradingSignal, Position
from finance.core import TradeAction, SignalSource, Timeframe
from datetime import datetime
from decimal import Decimal

# Market data
market_data = MarketData(
    symbol="AAPL",
    timestamp=datetime.now(),
    open=Decimal("150.00"),
    high=Decimal("152.50"),
    low=Decimal("149.80"),
    close=Decimal("151.20"),
    volume=1_000_000,
    provider="alpaca",
    timeframe=Timeframe.ONE_DAY
)

print(f"Typical price: {market_data.typical_price}")
print(f"Range: {market_data.price_range}")

# Trading signal
signal = TradingSignal(
    symbol="AAPL",
    action=TradeAction.BUY,
    confidence=0.85,
    timestamp=datetime.now(),
    source=SignalSource.ML,
    target_price=Decimal("155.00"),
    stop_loss=Decimal("148.00"),
    metadata={"model": "random_forest", "features": 42}
)

# Position tracking
position = Position(
    symbol="AAPL",
    quantity=100,  # Long 100 shares
    entry_price=Decimal("150.00"),
    current_price=Decimal("151.20")
)

print(f"Unrealized P&L: ${position.unrealized_pnl}")
print(f"P&L %: {position.unrealized_pnl_percent:.2f}%")

# Update position price
position.update_price(Decimal("152.00"))
```

### Metrics

```python
from finance.core import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    value_at_risk,
    win_rate,
    profit_factor
)

# Strategy returns (example: daily returns)
daily_returns = [0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.012]

# Performance metrics
sharpe = sharpe_ratio(daily_returns, risk_free_rate=0.02, periods_per_year=252)
print(f"Sharpe Ratio: {sharpe:.2f}")

sortino = sortino_ratio(daily_returns, risk_free_rate=0.02, periods_per_year=252)
print(f"Sortino Ratio: {sortino:.2f}")

max_dd = max_drawdown(daily_returns)
print(f"Max Drawdown: {max_dd:.2%}")

# Risk metrics
var_95 = value_at_risk(daily_returns, confidence_level=0.95)
print(f"95% VaR: {var_95:.2%}")

# Trade statistics
trade_pnl = [100, -50, 200, -30, 150, -40, 80]
win_pct = win_rate(trade_pnl)
pf = profit_factor(trade_pnl)

print(f"Win Rate: {win_pct:.1%}")
print(f"Profit Factor: {pf:.2f}")
```

## Module Reference

### `schemas.py`

**MarketData**: OHLCV data with validation
- Properties: `typical_price`, `price_range`
- Immutable for safety

**OptionsQuote**: Options contracts with Greeks
- Properties: `mid_price`, `spread`, `spread_percent`
- Greeks are optional (not all providers supply them)

**TradingSignal**: Canonical signal format
- Supports BUY/SELL/HOLD/SHORT/COVER actions
- Confidence scoring for position sizing
- Extensible metadata dictionary

**Position**: P&L tracking for open positions
- Supports long and short positions
- Properties: `unrealized_pnl`, `market_value`, `is_long`, `is_short`

**Enums**: `TradeAction`, `SignalSource`, `Timeframe`

### `metrics.py`

**Performance Metrics**:
- `sharpe_ratio()`: Risk-adjusted return (best for normally distributed returns)
- `sortino_ratio()`: Downside risk-adjusted return (penalizes only negative volatility)
- `calmar_ratio()`: Return-to-drawdown ratio (critical for drawdown-sensitive strategies)

**Risk Metrics**:
- `max_drawdown()`: Largest peak-to-trough decline
- `value_at_risk()`: Potential loss at confidence level (95%, 99%)
- `conditional_var()`: Expected loss beyond VaR (tail risk)
- `beta()`: Market sensitivity coefficient

**Trade Statistics**:
- `win_rate()`: Percentage of profitable trades
- `profit_factor()`: Gross profit / gross loss ratio
- `expectancy()`: Average expected profit per trade

## Contributing

### When to Add to Core Library

**DO add** if the code:
- Is used by 3+ agents
- Provides canonical data schemas
- Implements standard financial calculations
- Has stable interface (unlikely to change)

**DON'T add** if the code:
- Is agent-specific logic (e.g., ML model training)
- Requires agent-specific configuration
- Is experimental or rapidly changing
- Has external service dependencies (keep in agent code)

### Contribution Workflow

1. Add functionality to appropriate module (`metrics.py` or `schemas.py`)
2. Include type hints and comprehensive docstrings
3. Add input validation and error handling
4. Include test stubs as comments
5. Update `__init__.py` exports
6. Update this README with usage examples

### Code Standards

- **Type Safety**: Use type hints for all public functions
- **Validation**: Validate inputs, raise `ValueError` with clear messages
- **Documentation**: Google-style docstrings with "When to use" guidance
- **Precision**: Use `Decimal` for prices, `float` for ratios/metrics
- **Immutability**: Prefer frozen dataclasses for schemas (unless mutable by design)

## Testing

```bash
# Run tests (when implemented)
pytest tests/

# With coverage
pytest --cov=finance.core tests/
```

## Dependencies

- **pandas** (>=2.0.0): Time series analysis
- **numpy** (>=1.24.0): Numerical operations
- **scipy** (>=1.10.0): Statistical functions

## Version History

**1.0.0** (2025-10-02):
- Initial release
- Core schemas: MarketData, OptionsQuote, TradingSignal, Position
- Performance metrics: Sharpe, Sortino, Calmar, max drawdown
- Risk metrics: VaR, CVaR, beta
- Trade statistics: win rate, profit factor, expectancy

## License

MIT License - See project root for details.

## Support

For questions or issues, refer to the main Claude Agents Pro documentation or create an issue in the repository.
