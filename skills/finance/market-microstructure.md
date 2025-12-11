---
name: market-microstructure
description: Load when user needs order book analysis, bid-ask spread, market depth, liquidity metrics, price discovery, or market microstructure concepts
trigger_keywords: [order book, bid ask spread, market depth, liquidity, price discovery, market maker, tick size, queue position, level 2, market microstructure, book imbalance, mid price, microprice, trade flow, order flow, tob, top of book]
---

# Market Microstructure Skill

Production-grade market microstructure analysis including order book modeling, liquidity metrics, price discovery, and market impact estimation.

## Core Concepts

### Order Book Structure

```yaml
Level 1 (Top of Book):
  - Best Bid: Highest buy price
  - Best Ask: Lowest sell price
  - Bid Size: Quantity at best bid
  - Ask Size: Quantity at best ask
  - Spread: Ask - Bid

Level 2 (Depth of Book):
  - Multiple price levels on each side
  - Aggregate quantity at each level
  - Typically 5-20 levels

Level 3 (Full Order Book):
  - Individual orders at each price
  - Order IDs, timestamps
  - Queue position information
```

### Key Metrics

```yaml
Spread Metrics:
  Quoted Spread: Ask - Bid (absolute)
  Relative Spread: (Ask - Bid) / Mid (percentage)
  Effective Spread: 2 × |Trade Price - Mid| (actual cost)

Liquidity Metrics:
  Market Depth: Sum of quantity at N levels
  Book Imbalance: (Bid Qty - Ask Qty) / (Bid Qty + Ask Qty)
  Kyle's Lambda: Price impact per unit traded

Price Metrics:
  Mid Price: (Bid + Ask) / 2
  Microprice: Bid + (Ask - Bid) × (Bid Qty / (Bid Qty + Ask Qty))
  VWAP: Volume-weighted average price
```

## Implementation Patterns

### 1. Order Book Data Structures

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from sortedcontainers import SortedDict
import numpy as np

@dataclass
class Order:
    """Individual order in the book"""
    order_id: str
    price: float
    quantity: int
    side: str  # 'bid' or 'ask'
    timestamp: float

@dataclass
class PriceLevel:
    """Aggregate level in order book"""
    price: float
    total_quantity: int
    order_count: int
    orders: List[Order] = field(default_factory=list)

class OrderBook:
    """
    Limit Order Book with efficient updates

    Uses sorted dictionaries for O(log n) operations
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        # Bids sorted descending (highest first)
        self.bids: SortedDict = SortedDict(lambda x: -x)
        # Asks sorted ascending (lowest first)
        self.asks: SortedDict = SortedDict()
        self.orders: Dict[str, Order] = {}
        self.last_update: float = 0

    def add_order(self, order: Order):
        """Add new order to book"""
        book = self.bids if order.side == 'bid' else self.asks

        if order.price not in book:
            book[order.price] = PriceLevel(
                price=order.price,
                total_quantity=0,
                order_count=0
            )

        level = book[order.price]
        level.total_quantity += order.quantity
        level.order_count += 1
        level.orders.append(order)

        self.orders[order.order_id] = order
        self.last_update = order.timestamp

    def cancel_order(self, order_id: str):
        """Cancel order from book"""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        book = self.bids if order.side == 'bid' else self.asks

        if order.price in book:
            level = book[order.price]
            level.total_quantity -= order.quantity
            level.order_count -= 1
            level.orders = [o for o in level.orders if o.order_id != order_id]

            if level.total_quantity <= 0:
                del book[order.price]

        del self.orders[order_id]

    def modify_order(self, order_id: str, new_quantity: int):
        """Modify order quantity (loses queue priority)"""
        if order_id not in self.orders:
            return

        order = self.orders[order_id]
        self.cancel_order(order_id)

        if new_quantity > 0:
            new_order = Order(
                order_id=order_id,
                price=order.price,
                quantity=new_quantity,
                side=order.side,
                timestamp=order.timestamp
            )
            self.add_order(new_order)

    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Get best bid level"""
        if not self.bids:
            return None
        price = next(iter(self.bids))
        return self.bids[price]

    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Get best ask level"""
        if not self.asks:
            return None
        price = next(iter(self.asks))
        return self.asks[price]

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread"""
        if not self.best_bid or not self.best_ask:
            return None
        return self.best_ask.price - self.best_bid.price

    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        if not self.best_bid or not self.best_ask:
            return None
        return (self.best_bid.price + self.best_ask.price) / 2

    @property
    def microprice(self) -> Optional[float]:
        """
        Calculate microprice (size-weighted mid)

        Microprice = Bid + Spread × (Bid Size / (Bid Size + Ask Size))

        Better predictor of next trade price than mid
        """
        if not self.best_bid or not self.best_ask:
            return None

        bid_qty = self.best_bid.total_quantity
        ask_qty = self.best_ask.total_quantity
        total_qty = bid_qty + ask_qty

        if total_qty == 0:
            return self.mid_price

        imbalance = bid_qty / total_qty
        return self.best_bid.price + self.spread * imbalance

    def get_depth(self, side: str, levels: int = 5) -> List[PriceLevel]:
        """Get N levels of depth"""
        book = self.bids if side == 'bid' else self.asks
        return [book[price] for price in list(book.keys())[:levels]]

    def get_total_depth(self, side: str, levels: int = 5) -> int:
        """Get total quantity across N levels"""
        depth = self.get_depth(side, levels)
        return sum(level.total_quantity for level in depth)

    def simulate_market_order(self, side: str, quantity: int) -> Tuple[float, int]:
        """
        Simulate market order execution

        Args:
            side: 'buy' (hits asks) or 'sell' (hits bids)
            quantity: Order quantity

        Returns:
            (average_price, filled_quantity)
        """
        book = self.asks if side == 'buy' else self.bids
        remaining = quantity
        total_value = 0.0
        total_filled = 0

        for price in book:
            if remaining <= 0:
                break

            level = book[price]
            fill = min(remaining, level.total_quantity)

            total_value += fill * price
            total_filled += fill
            remaining -= fill

        if total_filled == 0:
            return 0.0, 0

        return total_value / total_filled, total_filled
```

### 2. Liquidity Metrics

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity analysis"""
    spread: float
    relative_spread_bps: float
    bid_depth: int
    ask_depth: int
    book_imbalance: float
    kyle_lambda: Optional[float]
    amihud_illiquidity: Optional[float]

class LiquidityAnalyzer:
    """Analyze order book liquidity"""

    def __init__(self, order_book: OrderBook):
        self.book = order_book

    def calculate_metrics(self, depth_levels: int = 5) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics"""
        mid = self.book.mid_price
        spread = self.book.spread

        if mid is None or spread is None:
            raise ValueError("Order book is empty")

        # Depth
        bid_depth = self.book.get_total_depth('bid', depth_levels)
        ask_depth = self.book.get_total_depth('ask', depth_levels)

        # Book imbalance: -1 (all asks) to +1 (all bids)
        total_depth = bid_depth + ask_depth
        imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0

        return LiquidityMetrics(
            spread=spread,
            relative_spread_bps=(spread / mid) * 10000,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            book_imbalance=imbalance,
            kyle_lambda=None,  # Requires trade data
            amihud_illiquidity=None  # Requires trade data
        )

    def estimate_market_impact(self, quantity: int, side: str) -> dict:
        """
        Estimate market impact of executing quantity

        Returns cost breakdown
        """
        mid = self.book.mid_price
        avg_price, filled = self.book.simulate_market_order(side, quantity)

        if filled == 0:
            return {'error': 'Insufficient liquidity'}

        # Calculate impact
        if side == 'buy':
            impact = (avg_price - mid) / mid
        else:
            impact = (mid - avg_price) / mid

        # Unfilled quantity
        unfilled = quantity - filled

        return {
            'average_price': avg_price,
            'filled_quantity': filled,
            'unfilled_quantity': unfilled,
            'market_impact_bps': impact * 10000,
            'execution_cost': abs(avg_price - mid) * filled,
            'fill_rate': filled / quantity
        }

    def calculate_effective_spread(
        self,
        trades: List[dict]  # List of {'price': float, 'quantity': int, 'side': str}
    ) -> float:
        """
        Calculate effective spread from trade data

        Effective Spread = 2 × |Trade Price - Mid at trade time|
        Better measure of actual trading cost than quoted spread
        """
        if not trades:
            return 0.0

        mid = self.book.mid_price
        total_value = 0.0
        total_qty = 0

        for trade in trades:
            effective = 2 * abs(trade['price'] - mid)
            total_value += effective * trade['quantity']
            total_qty += trade['quantity']

        return total_value / total_qty if total_qty > 0 else 0.0


def calculate_kyle_lambda(
    price_changes: np.ndarray,
    signed_volumes: np.ndarray
) -> float:
    """
    Calculate Kyle's Lambda (price impact coefficient)

    ΔP = λ × SignedVolume + ε

    λ measures how much price moves per unit of signed order flow
    Higher λ = less liquid market
    """
    # OLS regression
    # λ = Cov(ΔP, V) / Var(V)
    covariance = np.cov(price_changes, signed_volumes)[0, 1]
    variance = np.var(signed_volumes)

    if variance == 0:
        return 0.0

    return covariance / variance


def calculate_amihud_illiquidity(
    returns: np.ndarray,
    volumes: np.ndarray
) -> float:
    """
    Amihud Illiquidity Ratio

    ILLIQ = Average(|Return| / Dollar Volume)

    Higher = more illiquid (price moves more per unit volume)
    """
    dollar_volumes = volumes  # Assume already in dollar terms
    valid_idx = dollar_volumes > 0

    illiq = np.mean(np.abs(returns[valid_idx]) / dollar_volumes[valid_idx])
    return illiq
```

### 3. Price Discovery Metrics

```python
from typing import List, Tuple
import numpy as np
from scipy.stats import linregress

def calculate_price_contribution(
    venue_prices: Dict[str, np.ndarray],
    timestamps: np.ndarray
) -> Dict[str, float]:
    """
    Calculate price discovery contribution by venue

    Uses Hasbrouck Information Share methodology
    Higher share = venue contributes more to price discovery
    """
    # Simplified version - in practice use VAR model
    venues = list(venue_prices.keys())
    n_venues = len(venues)

    # Calculate returns for each venue
    returns = {v: np.diff(np.log(venue_prices[v])) for v in venues}

    # Calculate variance contributions
    total_variance = np.var(sum(returns.values()))
    contributions = {}

    for venue in venues:
        venue_var = np.var(returns[venue])
        contributions[venue] = venue_var / total_variance if total_variance > 0 else 1/n_venues

    return contributions


def calculate_quote_midpoint_variance(
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    intervals: List[int] = [1, 5, 10, 30, 60]
) -> Dict[int, float]:
    """
    Calculate variance of quote midpoint at different time scales

    Used to assess price efficiency and mean reversion
    """
    mid_prices = (bid_prices + ask_prices) / 2
    log_mid = np.log(mid_prices)

    variances = {}
    for interval in intervals:
        if len(log_mid) > interval:
            returns = log_mid[interval:] - log_mid[:-interval]
            variances[interval] = np.var(returns)

    return variances


def detect_quote_stuffing(
    quotes: List[dict],  # {'timestamp': float, 'bid': float, 'ask': float}
    window_seconds: float = 1.0,
    threshold_quotes_per_second: int = 100
) -> List[Tuple[float, float]]:
    """
    Detect potential quote stuffing (excessive quote updates)

    Returns time ranges with suspicious activity
    """
    suspicious_periods = []

    if not quotes:
        return suspicious_periods

    # Count quotes per window
    window_start = quotes[0]['timestamp']
    quote_count = 0

    for quote in quotes:
        if quote['timestamp'] - window_start <= window_seconds:
            quote_count += 1
        else:
            # Check if suspicious
            if quote_count / window_seconds > threshold_quotes_per_second:
                suspicious_periods.append((window_start, window_start + window_seconds))

            # Move window
            window_start = quote['timestamp']
            quote_count = 1

    return suspicious_periods
```

### 4. Trade Flow Analysis

```python
from enum import Enum
from typing import List, Tuple
import numpy as np

class TradeClassification(Enum):
    BUY = 1
    SELL = -1
    UNKNOWN = 0

def classify_trade_lee_ready(
    trade_price: float,
    bid: float,
    ask: float,
    prev_trade_price: Optional[float] = None
) -> TradeClassification:
    """
    Lee-Ready trade classification algorithm

    1. If trade price > mid: buyer-initiated
    2. If trade price < mid: seller-initiated
    3. If at mid: use tick test (compare to previous trade)
    """
    mid = (bid + ask) / 2

    if trade_price > mid:
        return TradeClassification.BUY
    elif trade_price < mid:
        return TradeClassification.SELL
    else:
        # Tick test for trades at mid
        if prev_trade_price is not None:
            if trade_price > prev_trade_price:
                return TradeClassification.BUY
            elif trade_price < prev_trade_price:
                return TradeClassification.SELL

        return TradeClassification.UNKNOWN


def calculate_order_flow_imbalance(
    trades: List[dict],  # {'price': float, 'quantity': int, 'bid': float, 'ask': float}
    window_size: int = 100
) -> np.ndarray:
    """
    Calculate rolling order flow imbalance

    OFI = Σ(Buy Volume) - Σ(Sell Volume)

    Positive OFI = net buying pressure
    Negative OFI = net selling pressure
    """
    signed_volumes = []
    prev_price = None

    for trade in trades:
        classification = classify_trade_lee_ready(
            trade['price'],
            trade['bid'],
            trade['ask'],
            prev_price
        )

        signed_vol = trade['quantity'] * classification.value
        signed_volumes.append(signed_vol)
        prev_price = trade['price']

    # Rolling sum
    signed_volumes = np.array(signed_volumes)
    ofi = np.convolve(signed_volumes, np.ones(window_size), mode='valid')

    return ofi


def calculate_vpin(
    trades: List[dict],
    bucket_size: int = 10000,  # Volume per bucket
    n_buckets: int = 50
) -> float:
    """
    Volume-Synchronized Probability of Informed Trading (VPIN)

    Higher VPIN indicates higher probability of informed trading
    and potential for adverse selection

    Returns VPIN between 0 and 1
    """
    # Classify trades
    buy_volume = 0
    sell_volume = 0
    bucket_imbalances = []
    current_bucket_volume = 0
    prev_price = None

    for trade in trades:
        classification = classify_trade_lee_ready(
            trade['price'],
            trade['bid'],
            trade['ask'],
            prev_price
        )

        qty = trade['quantity']

        if classification == TradeClassification.BUY:
            buy_volume += qty
        elif classification == TradeClassification.SELL:
            sell_volume += qty
        else:
            # Split unknown trades
            buy_volume += qty / 2
            sell_volume += qty / 2

        current_bucket_volume += qty
        prev_price = trade['price']

        # Check if bucket is complete
        if current_bucket_volume >= bucket_size:
            imbalance = abs(buy_volume - sell_volume) / (buy_volume + sell_volume)
            bucket_imbalances.append(imbalance)

            # Reset for next bucket
            buy_volume = 0
            sell_volume = 0
            current_bucket_volume = 0

            if len(bucket_imbalances) >= n_buckets:
                break

    if len(bucket_imbalances) < n_buckets:
        return 0.0

    # VPIN = Average of bucket imbalances
    return np.mean(bucket_imbalances[-n_buckets:])
```

## Production Order Book Processor

```python
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from collections import deque
import time

@dataclass
class MarketUpdate:
    """Real-time market data update"""
    timestamp: float
    update_type: str  # 'quote', 'trade', 'book_update'
    symbol: str
    data: dict

class OrderBookProcessor:
    """
    Production order book processor with real-time analytics

    Features:
    - Efficient incremental updates
    - Rolling liquidity metrics
    - Trade flow analysis
    - Alerting on anomalies
    """

    def __init__(
        self,
        symbol: str,
        metrics_window: int = 1000,
        alert_callback: Optional[Callable] = None
    ):
        self.symbol = symbol
        self.book = OrderBook(symbol)
        self.analyzer = LiquidityAnalyzer(self.book)

        # Rolling windows
        self.recent_trades: deque = deque(maxlen=metrics_window)
        self.recent_spreads: deque = deque(maxlen=metrics_window)
        self.recent_imbalances: deque = deque(maxlen=metrics_window)

        # Alerting
        self.alert_callback = alert_callback
        self.spread_threshold_bps = 50  # Alert if spread > 50 bps
        self.imbalance_threshold = 0.8  # Alert if |imbalance| > 0.8

    def process_update(self, update: MarketUpdate):
        """Process incoming market data update"""
        if update.update_type == 'book_update':
            self._process_book_update(update.data)
        elif update.update_type == 'trade':
            self._process_trade(update.data)
        elif update.update_type == 'quote':
            self._process_quote(update.data)

        # Update rolling metrics
        self._update_metrics()

        # Check for alerts
        self._check_alerts()

    def _process_book_update(self, data: dict):
        """Process order book update (add/modify/cancel)"""
        action = data.get('action')

        if action == 'add':
            order = Order(
                order_id=data['order_id'],
                price=data['price'],
                quantity=data['quantity'],
                side=data['side'],
                timestamp=time.time()
            )
            self.book.add_order(order)
        elif action == 'cancel':
            self.book.cancel_order(data['order_id'])
        elif action == 'modify':
            self.book.modify_order(data['order_id'], data['quantity'])

    def _process_trade(self, data: dict):
        """Process trade event"""
        self.recent_trades.append({
            'timestamp': time.time(),
            'price': data['price'],
            'quantity': data['quantity'],
            'bid': self.book.best_bid.price if self.book.best_bid else data['price'],
            'ask': self.book.best_ask.price if self.book.best_ask else data['price']
        })

    def _process_quote(self, data: dict):
        """Process quote update (top of book)"""
        # Update book with new top-of-book quotes
        # This is a simplified version - production would handle full depth
        pass

    def _update_metrics(self):
        """Update rolling metrics"""
        if self.book.spread is not None and self.book.mid_price is not None:
            spread_bps = (self.book.spread / self.book.mid_price) * 10000
            self.recent_spreads.append(spread_bps)

        if self.book.best_bid and self.book.best_ask:
            bid_qty = self.book.best_bid.total_quantity
            ask_qty = self.book.best_ask.total_quantity
            total = bid_qty + ask_qty
            if total > 0:
                imbalance = (bid_qty - ask_qty) / total
                self.recent_imbalances.append(imbalance)

    def _check_alerts(self):
        """Check for alert conditions"""
        if not self.alert_callback:
            return

        # Spread alert
        if self.recent_spreads and self.recent_spreads[-1] > self.spread_threshold_bps:
            self.alert_callback({
                'type': 'wide_spread',
                'symbol': self.symbol,
                'spread_bps': self.recent_spreads[-1],
                'threshold': self.spread_threshold_bps
            })

        # Imbalance alert
        if self.recent_imbalances and abs(self.recent_imbalances[-1]) > self.imbalance_threshold:
            self.alert_callback({
                'type': 'extreme_imbalance',
                'symbol': self.symbol,
                'imbalance': self.recent_imbalances[-1],
                'threshold': self.imbalance_threshold
            })

    def get_current_state(self) -> dict:
        """Get current order book state and metrics"""
        try:
            metrics = self.analyzer.calculate_metrics()
        except ValueError:
            metrics = None

        return {
            'symbol': self.symbol,
            'timestamp': time.time(),
            'best_bid': self.book.best_bid.price if self.book.best_bid else None,
            'best_ask': self.book.best_ask.price if self.book.best_ask else None,
            'mid_price': self.book.mid_price,
            'microprice': self.book.microprice,
            'spread': self.book.spread,
            'metrics': metrics,
            'avg_spread_bps': np.mean(self.recent_spreads) if self.recent_spreads else None,
            'avg_imbalance': np.mean(self.recent_imbalances) if self.recent_imbalances else None,
            'trade_count': len(self.recent_trades)
        }


# Usage Example
if __name__ == "__main__":
    def alert_handler(alert):
        print(f"ALERT: {alert['type']} - {alert}")

    processor = OrderBookProcessor(
        symbol="AAPL",
        alert_callback=alert_handler
    )

    # Simulate order book updates
    processor.process_update(MarketUpdate(
        timestamp=time.time(),
        update_type='book_update',
        symbol='AAPL',
        data={
            'action': 'add',
            'order_id': 'O001',
            'price': 150.00,
            'quantity': 1000,
            'side': 'bid'
        }
    ))

    processor.process_update(MarketUpdate(
        timestamp=time.time(),
        update_type='book_update',
        symbol='AAPL',
        data={
            'action': 'add',
            'order_id': 'O002',
            'price': 150.05,
            'quantity': 500,
            'side': 'ask'
        }
    ))

    state = processor.get_current_state()
    print(f"Mid Price: ${state['mid_price']:.2f}")
    print(f"Spread: ${state['spread']:.2f}")
    print(f"Microprice: ${state['microprice']:.4f}")
```

## Best Practices

1. **Use efficient data structures** (sorted containers) for O(log n) operations
2. **Track microprice** as better fair value estimate than mid price
3. **Monitor multiple liquidity metrics** - no single metric tells full story
4. **Implement queue position tracking** for execution quality analysis
5. **Use VPIN** for informed trading probability estimation
6. **Handle crossed/locked markets** gracefully in production

## Common Pitfalls

❌ **Ignoring queue position** when estimating fill probability
✅ Track and model queue dynamics

❌ **Using mid price** when microprice is more accurate
✅ Use microprice for fairer value estimation

❌ **Not handling stale quotes** in fast markets
✅ Implement quote staleness detection and handling

❌ **Assuming symmetric book** when calculating impact
✅ Model bid/ask sides separately

❌ **Ignoring hidden liquidity** (iceberg orders)
✅ Account for hidden order detection heuristics

## Quality Standards

- **Latency**: Process updates in <1ms
- **Memory**: O(n) where n = total orders in book
- **Accuracy**: Microprice within tick size of true fair value
- **Throughput**: Handle >10,000 updates/second
- **Recovery**: Rebuild book from snapshot in <100ms

---

**Skill Type**: Finance - Market Microstructure
**Complexity**: Complex
**Typical Usage**: Activated when market-data-engineer or algorithmic-trading-engineer needs order book analysis
**Performance**: Real-time processing with sub-millisecond latency
