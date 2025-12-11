---
name: algo-trading-systems
description: Load when user needs algorithmic trading system design, trading infrastructure, strategy patterns, or live trading architecture. Covers complete data-to-execution pipeline.
trigger_keywords: [algo trading, algorithmic trading, trading system, trading bot, automated trading, quant trading, systematic trading, trading infrastructure, trading engine, alpha generation, signal generation, live trading, paper trading, trading architecture, event-driven trading, strategy pattern, momentum strategy, mean reversion, pairs trading, stat arb]
---

# Algorithmic Trading Systems Skill

Complete guide to building production algorithmic trading systems from data ingestion to live execution.

## System Architecture

### High-Level Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    ALGO TRADING SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │   Data   │ → │  Signal  │ → │ Position │ → │Execution │    │
│  │ Pipeline │   │Generator │   │  Sizer   │   │  Engine  │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       ↑                                              ↓         │
│  ┌──────────┐                               ┌──────────┐       │
│  │  Market  │                               │  Broker  │       │
│  │   Data   │                               │   API    │       │
│  └──────────┘                               └──────────┘       │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Risk Manager (Position Limits, P&L)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           Monitoring & Alerting (Grafana/PagerDuty)      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Core Classes

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
import asyncio

class Side(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Signal:
    """Trading signal from strategy"""
    symbol: str
    side: Side
    strength: float      # -1.0 to 1.0
    timestamp: datetime
    strategy_id: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class Order:
    """Order to be executed"""
    id: str
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0
    avg_fill_price: float = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class Position:
    """Current position in an instrument"""
    symbol: str
    quantity: float
    avg_cost: float
    market_price: float
    unrealized_pnl: float = 0
    realized_pnl: float = 0

    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price

    @property
    def total_pnl(self) -> float:
        return self.unrealized_pnl + self.realized_pnl
```

## Strategy Patterns

### Base Strategy Interface

```python
class Strategy(ABC):
    """Base class for all trading strategies"""

    def __init__(self, strategy_id: str, symbols: List[str]):
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.positions: Dict[str, Position] = {}
        self.signals: List[Signal] = []

    @abstractmethod
    def on_bar(self, symbol: str, bar: dict) -> Optional[Signal]:
        """Called on each new bar - return signal or None"""
        pass

    @abstractmethod
    def on_tick(self, symbol: str, tick: dict) -> Optional[Signal]:
        """Called on each tick - return signal or None"""
        pass

    def on_fill(self, order: Order):
        """Called when order is filled - update internal state"""
        pass

    def on_start(self):
        """Called when strategy starts"""
        pass

    def on_stop(self):
        """Called when strategy stops"""
        pass
```

### Momentum Strategy

```python
import numpy as np
from collections import deque

class MomentumStrategy(Strategy):
    """
    Trend-following momentum strategy

    Entry: Price breaks above/below N-period high/low
    Exit: Trailing stop or reversal signal
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        lookback: int = 20,
        atr_multiplier: float = 2.0
    ):
        super().__init__(strategy_id, symbols)
        self.lookback = lookback
        self.atr_multiplier = atr_multiplier
        self.price_history: Dict[str, deque] = {
            s: deque(maxlen=lookback) for s in symbols
        }
        self.atr: Dict[str, float] = {}

    def on_bar(self, symbol: str, bar: dict) -> Optional[Signal]:
        self.price_history[symbol].append(bar)

        if len(self.price_history[symbol]) < self.lookback:
            return None

        closes = [b['close'] for b in self.price_history[symbol]]
        highs = [b['high'] for b in self.price_history[symbol]]
        lows = [b['low'] for b in self.price_history[symbol]]

        # Calculate breakout levels
        highest_high = max(highs[:-1])  # Exclude current bar
        lowest_low = min(lows[:-1])
        current_close = closes[-1]

        # ATR for position sizing
        self.atr[symbol] = self._calculate_atr(highs, lows, closes)

        # Generate signals
        if current_close > highest_high:
            return Signal(
                symbol=symbol,
                side=Side.BUY,
                strength=1.0,
                timestamp=datetime.utcnow(),
                strategy_id=self.strategy_id,
                metadata={'breakout_level': highest_high, 'atr': self.atr[symbol]}
            )
        elif current_close < lowest_low:
            return Signal(
                symbol=symbol,
                side=Side.SELL,
                strength=-1.0,
                timestamp=datetime.utcnow(),
                strategy_id=self.strategy_id,
                metadata={'breakout_level': lowest_low, 'atr': self.atr[symbol]}
            )

        return None

    def _calculate_atr(self, highs, lows, closes, period=14) -> float:
        """Average True Range"""
        trs = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            )
            trs.append(tr)
        return np.mean(trs[-period:]) if len(trs) >= period else np.mean(trs)
```

### Mean Reversion Strategy

```python
class MeanReversionStrategy(Strategy):
    """
    Mean reversion using Bollinger Bands

    Entry: Price touches lower/upper band
    Exit: Price returns to middle band
    """

    def __init__(
        self,
        strategy_id: str,
        symbols: List[str],
        lookback: int = 20,
        num_std: float = 2.0
    ):
        super().__init__(strategy_id, symbols)
        self.lookback = lookback
        self.num_std = num_std
        self.price_history: Dict[str, deque] = {
            s: deque(maxlen=lookback) for s in symbols
        }

    def on_bar(self, symbol: str, bar: dict) -> Optional[Signal]:
        self.price_history[symbol].append(bar['close'])

        if len(self.price_history[symbol]) < self.lookback:
            return None

        prices = list(self.price_history[symbol])
        sma = np.mean(prices)
        std = np.std(prices)

        upper_band = sma + self.num_std * std
        lower_band = sma - self.num_std * std
        current_price = prices[-1]

        # Z-score for signal strength
        z_score = (current_price - sma) / std if std > 0 else 0

        # Mean reversion signals
        if current_price <= lower_band:
            return Signal(
                symbol=symbol,
                side=Side.BUY,
                strength=min(abs(z_score) / 3, 1.0),  # Normalize strength
                timestamp=datetime.utcnow(),
                strategy_id=self.strategy_id,
                metadata={'z_score': z_score, 'sma': sma}
            )
        elif current_price >= upper_band:
            return Signal(
                symbol=symbol,
                side=Side.SELL,
                strength=-min(abs(z_score) / 3, 1.0),
                timestamp=datetime.utcnow(),
                strategy_id=self.strategy_id,
                metadata={'z_score': z_score, 'sma': sma}
            )

        return None
```

### Pairs Trading / Statistical Arbitrage

```python
class PairsTradingStrategy(Strategy):
    """
    Statistical arbitrage on correlated pairs

    Entry: Spread deviates > threshold from mean
    Exit: Spread reverts to mean
    """

    def __init__(
        self,
        strategy_id: str,
        symbol_a: str,
        symbol_b: str,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5
    ):
        super().__init__(strategy_id, [symbol_a, symbol_b])
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.prices_a: deque = deque(maxlen=lookback)
        self.prices_b: deque = deque(maxlen=lookback)
        self.hedge_ratio: float = 1.0
        self.in_position: bool = False
        self.position_side: Optional[str] = None

    def on_bar(self, symbol: str, bar: dict) -> Optional[Signal]:
        if symbol == self.symbol_a:
            self.prices_a.append(bar['close'])
        else:
            self.prices_b.append(bar['close'])

        if len(self.prices_a) < self.lookback or len(self.prices_b) < self.lookback:
            return None

        # Calculate hedge ratio via OLS
        prices_a = np.array(self.prices_a)
        prices_b = np.array(self.prices_b)
        self.hedge_ratio = np.cov(prices_a, prices_b)[0, 1] / np.var(prices_b)

        # Calculate spread
        spread = prices_a - self.hedge_ratio * prices_b
        spread_mean = np.mean(spread)
        spread_std = np.std(spread)
        current_spread = spread[-1]
        z_score = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0

        # Generate signals
        if not self.in_position:
            if z_score > self.entry_z:
                # Spread too high: short A, long B
                self.in_position = True
                self.position_side = 'short_spread'
                return Signal(
                    symbol=self.symbol_a,
                    side=Side.SELL,
                    strength=-1.0,
                    timestamp=datetime.utcnow(),
                    strategy_id=self.strategy_id,
                    metadata={'z_score': z_score, 'hedge_ratio': self.hedge_ratio, 'pair': 'short_spread'}
                )
            elif z_score < -self.entry_z:
                # Spread too low: long A, short B
                self.in_position = True
                self.position_side = 'long_spread'
                return Signal(
                    symbol=self.symbol_a,
                    side=Side.BUY,
                    strength=1.0,
                    timestamp=datetime.utcnow(),
                    strategy_id=self.strategy_id,
                    metadata={'z_score': z_score, 'hedge_ratio': self.hedge_ratio, 'pair': 'long_spread'}
                )
        else:
            # Exit when spread reverts
            if abs(z_score) < self.exit_z:
                self.in_position = False
                exit_side = Side.BUY if self.position_side == 'short_spread' else Side.SELL
                return Signal(
                    symbol=self.symbol_a,
                    side=exit_side,
                    strength=0.0,  # Exit signal
                    timestamp=datetime.utcnow(),
                    strategy_id=self.strategy_id,
                    metadata={'z_score': z_score, 'action': 'exit'}
                )

        return None
```

## Position Sizing

```python
class PositionSizer:
    """Calculate position sizes based on risk parameters"""

    def __init__(
        self,
        account_equity: float,
        max_position_pct: float = 0.10,    # Max 10% per position
        max_risk_per_trade: float = 0.02,  # Risk 2% per trade
        max_total_exposure: float = 1.0     # 100% max exposure
    ):
        self.account_equity = account_equity
        self.max_position_pct = max_position_pct
        self.max_risk_per_trade = max_risk_per_trade
        self.max_total_exposure = max_total_exposure

    def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        stop_loss_price: float,
        current_exposure: float = 0
    ) -> float:
        """
        Calculate position size using risk-based sizing

        Position Size = (Account Risk) / (Per-Share Risk)
        """
        # Risk per share
        risk_per_share = abs(current_price - stop_loss_price)
        if risk_per_share == 0:
            return 0

        # Maximum dollar risk
        dollar_risk = self.account_equity * self.max_risk_per_trade

        # Risk-based size
        risk_based_size = dollar_risk / risk_per_share

        # Position limit based on account %
        max_position_value = self.account_equity * self.max_position_pct
        position_limit_size = max_position_value / current_price

        # Exposure limit
        remaining_exposure = self.max_total_exposure - current_exposure
        exposure_limit_value = self.account_equity * remaining_exposure
        exposure_limit_size = exposure_limit_value / current_price

        # Take minimum of all constraints
        position_size = min(
            risk_based_size,
            position_limit_size,
            exposure_limit_size
        )

        # Scale by signal strength
        position_size *= abs(signal.strength)

        return max(0, int(position_size))

    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Kelly Criterion for optimal bet sizing

        f* = (p * b - q) / b
        Where: p = win rate, q = 1-p, b = win/loss ratio
        """
        if avg_loss == 0:
            return 0

        b = avg_win / avg_loss
        q = 1 - win_rate
        kelly = (win_rate * b - q) / b

        # Use fractional Kelly (half) for safety
        return max(0, min(kelly * 0.5, 0.25))
```

## Execution Engine

```python
class ExecutionEngine:
    """Manages order execution and broker communication"""

    def __init__(self, broker_api, max_slippage: float = 0.001):
        self.broker = broker_api
        self.max_slippage = max_slippage
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []

    async def execute_signal(
        self,
        signal: Signal,
        quantity: float,
        order_type: OrderType = OrderType.LIMIT
    ) -> Order:
        """Convert signal to order and execute"""

        # Get current market price
        quote = await self.broker.get_quote(signal.symbol)
        price = quote['ask'] if signal.side == Side.BUY else quote['bid']

        # Create order
        order = Order(
            id=self._generate_order_id(),
            symbol=signal.symbol,
            side=signal.side,
            quantity=quantity,
            order_type=order_type,
            limit_price=price if order_type == OrderType.LIMIT else None
        )

        # Submit to broker
        try:
            response = await self.broker.submit_order(order)
            order.status = OrderStatus.SUBMITTED
            self.pending_orders[order.id] = order
        except Exception as e:
            order.status = OrderStatus.REJECTED
            order.metadata['reject_reason'] = str(e)

        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self.pending_orders:
            await self.broker.cancel_order(order_id)
            self.pending_orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False

    def on_fill(self, fill_event: dict):
        """Handle fill notification from broker"""
        order_id = fill_event['order_id']
        if order_id in self.pending_orders:
            order = self.pending_orders[order_id]
            order.filled_qty += fill_event['quantity']
            order.avg_fill_price = fill_event['price']

            if order.filled_qty >= order.quantity:
                order.status = OrderStatus.FILLED
                del self.pending_orders[order_id]
                self.order_history.append(order)
```

## Risk Manager

```python
class RiskManager:
    """Real-time risk monitoring and controls"""

    def __init__(
        self,
        max_daily_loss: float,
        max_position_size: float,
        max_drawdown: float,
        position_limits: Dict[str, float] = None
    ):
        self.max_daily_loss = max_daily_loss
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.position_limits = position_limits or {}

        self.daily_pnl: float = 0
        self.peak_equity: float = 0
        self.current_equity: float = 0
        self.is_halted: bool = False
        self.halt_reason: str = ""

    def check_order(self, order: Order, positions: Dict[str, Position]) -> tuple[bool, str]:
        """Pre-trade risk check"""

        if self.is_halted:
            return False, f"Trading halted: {self.halt_reason}"

        # Position limit check
        symbol = order.symbol
        current_qty = positions.get(symbol, Position(symbol, 0, 0, 0)).quantity
        new_qty = current_qty + (order.quantity if order.side == Side.BUY else -order.quantity)

        if abs(new_qty) > self.max_position_size:
            return False, f"Position limit exceeded: {new_qty} > {self.max_position_size}"

        # Symbol-specific limits
        if symbol in self.position_limits:
            if abs(new_qty) > self.position_limits[symbol]:
                return False, f"Symbol limit exceeded for {symbol}"

        return True, "OK"

    def update_pnl(self, pnl_change: float, current_equity: float):
        """Update P&L and check limits"""
        self.daily_pnl += pnl_change
        self.current_equity = current_equity
        self.peak_equity = max(self.peak_equity, current_equity)

        # Daily loss limit
        if self.daily_pnl < -self.max_daily_loss:
            self.halt_trading(f"Daily loss limit hit: {self.daily_pnl}")

        # Drawdown limit
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.halt_trading(f"Max drawdown exceeded: {drawdown:.2%}")

    def halt_trading(self, reason: str):
        """Emergency halt"""
        self.is_halted = True
        self.halt_reason = reason
        # Trigger alerts, flatten positions if needed

    def reset_daily(self):
        """Reset daily counters"""
        self.daily_pnl = 0
```

## Event-Driven Engine

```python
class TradingEngine:
    """Main event loop coordinating all components"""

    def __init__(
        self,
        strategies: List[Strategy],
        position_sizer: PositionSizer,
        execution_engine: ExecutionEngine,
        risk_manager: RiskManager,
        data_feed
    ):
        self.strategies = strategies
        self.position_sizer = position_sizer
        self.execution = execution_engine
        self.risk = risk_manager
        self.data_feed = data_feed

        self.positions: Dict[str, Position] = {}
        self.is_running: bool = False

    async def run(self):
        """Main event loop"""
        self.is_running = True

        for strategy in self.strategies:
            strategy.on_start()

        async for event in self.data_feed.stream():
            if not self.is_running:
                break

            await self._handle_event(event)

    async def _handle_event(self, event: dict):
        """Route events to appropriate handlers"""
        event_type = event.get('type')

        if event_type == 'bar':
            await self._handle_bar(event)
        elif event_type == 'tick':
            await self._handle_tick(event)
        elif event_type == 'fill':
            await self._handle_fill(event)
        elif event_type == 'error':
            await self._handle_error(event)

    async def _handle_bar(self, event: dict):
        """Process bar data through strategies"""
        symbol = event['symbol']
        bar = event['data']

        for strategy in self.strategies:
            if symbol in strategy.symbols:
                signal = strategy.on_bar(symbol, bar)

                if signal:
                    await self._process_signal(signal, bar['close'])

    async def _process_signal(self, signal: Signal, current_price: float):
        """Convert signal to order after risk checks"""

        # Calculate position size
        stop_loss = current_price * (0.98 if signal.side == Side.BUY else 1.02)
        quantity = self.position_sizer.calculate_position_size(
            signal=signal,
            current_price=current_price,
            stop_loss_price=stop_loss
        )

        if quantity == 0:
            return

        # Create tentative order
        order = Order(
            id="temp",
            symbol=signal.symbol,
            side=signal.side,
            quantity=quantity,
            order_type=OrderType.LIMIT,
            limit_price=current_price
        )

        # Risk check
        approved, reason = self.risk.check_order(order, self.positions)
        if not approved:
            print(f"Order rejected: {reason}")
            return

        # Execute
        await self.execution.execute_signal(signal, quantity)

    def stop(self):
        """Graceful shutdown"""
        self.is_running = False
        for strategy in self.strategies:
            strategy.on_stop()
```

## Strategy Lifecycle

```
┌────────────────────────────────────────────────────────────────┐
│                    STRATEGY LIFECYCLE                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. RESEARCH        2. BACKTEST         3. PAPER TRADE        │
│  ┌──────────┐      ┌──────────┐        ┌──────────┐          │
│  │ Ideation │  →   │Historical│   →    │ Live Data│          │
│  │ Analysis │      │Simulation│        │ Sim Exec │          │
│  └──────────┘      └──────────┘        └──────────┘          │
│                                              ↓                 │
│  6. MONITOR         5. SCALE            4. LIVE SMALL         │
│  ┌──────────┐      ┌──────────┐        ┌──────────┐          │
│  │ Metrics  │  ←   │ Increase │   ←    │Real Money│          │
│  │ Alerts   │      │ Capital  │        │Small Size│          │
│  └──────────┘      └──────────┘        └──────────┘          │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Gate Criteria:
  Research → Backtest: Clear hypothesis, defined rules
  Backtest → Paper: Sharpe > 1.0, Win rate > 45%, Max DD < 20%
  Paper → Live: 30+ days profitable, matches backtest within 20%
  Live Small → Scale: 90 days track record, risk-adjusted returns
```

## Production Monitoring

```python
from dataclasses import dataclass
from typing import Dict
import time

@dataclass
class TradingMetrics:
    """Real-time trading metrics"""
    total_pnl: float = 0
    daily_pnl: float = 0
    win_rate: float = 0
    sharpe_ratio: float = 0
    max_drawdown: float = 0
    current_drawdown: float = 0
    total_trades: int = 0
    winning_trades: int = 0
    avg_win: float = 0
    avg_loss: float = 0
    exposure: float = 0

class MetricsCollector:
    """Collect and export trading metrics"""

    def __init__(self):
        self.metrics = TradingMetrics()
        self.trade_history: List[dict] = []

    def record_trade(self, pnl: float, side: str):
        """Record completed trade"""
        self.metrics.total_trades += 1
        self.metrics.total_pnl += pnl
        self.metrics.daily_pnl += pnl

        if pnl > 0:
            self.metrics.winning_trades += 1

        self.metrics.win_rate = self.metrics.winning_trades / self.metrics.total_trades

        self.trade_history.append({
            'timestamp': time.time(),
            'pnl': pnl,
            'side': side
        })

    def export_prometheus(self) -> Dict[str, float]:
        """Export for Prometheus/Grafana"""
        return {
            'trading_pnl_total': self.metrics.total_pnl,
            'trading_pnl_daily': self.metrics.daily_pnl,
            'trading_win_rate': self.metrics.win_rate,
            'trading_sharpe': self.metrics.sharpe_ratio,
            'trading_drawdown': self.metrics.current_drawdown,
            'trading_trades_total': self.metrics.total_trades,
            'trading_exposure': self.metrics.exposure
        }
```

## Best Practices

1. **Start simple**: Single strategy, single instrument, paper trade first
2. **Risk first**: Position sizing and risk limits before strategy logic
3. **Log everything**: Every signal, order, fill, error for debugging
4. **Expect failures**: Network errors, API limits, data gaps - handle gracefully
5. **Monitor latency**: Track order-to-fill time, data freshness
6. **Test edge cases**: Market open/close, halts, splits, dividends

## Common Pitfalls

- **Overfitting**: In-sample results don't translate to live
- **Ignoring costs**: Commissions, slippage, market impact destroy edge
- **Curve fitting**: Too many parameters, too little data
- **Survivorship bias**: Testing only on current universe
- **Execution gap**: Backtest assumes fills that don't happen live
- **Over-leverage**: One bad day wipes out months of gains

---

**Skill Type**: Finance - Algorithmic Trading
**Complexity**: Advanced
**Typical Usage**: Building complete trading systems from data to execution
**Related Skills**: order-execution, backtesting-patterns, technical-indicators, risk-metrics
