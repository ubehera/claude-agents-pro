---
name: order-execution
description: Load when user needs TWAP, VWAP, iceberg orders, execution algorithms, order management, slippage modeling, or trade execution strategies
trigger_keywords: [twap, vwap, iceberg, execution algorithm, order execution, slippage, market impact, execution cost, smart order routing, order management, oms, fill rate, arrival price, implementation shortfall, pov, participation rate]
---

# Order Execution Algorithms Skill

Production-grade execution algorithms including TWAP, VWAP, POV, and iceberg orders with market impact modeling and slippage estimation.

## Core Concepts

### Execution Goals

**Primary Objectives**:
- Minimize market impact (price movement caused by order)
- Reduce information leakage (hiding order intent)
- Achieve target benchmark (VWAP, arrival price, close)
- Balance urgency vs. cost

### Execution Benchmarks

```yaml
Arrival Price:
  Definition: Price when order first entered
  Use Case: Aggressive execution, momentum trades
  Metric: Implementation Shortfall = Arrival - Avg Fill

VWAP (Volume-Weighted Average Price):
  Definition: Sum(Price × Volume) / Total Volume
  Use Case: Passive execution, reduce impact
  Metric: VWAP Slippage = VWAP - Avg Fill

TWAP (Time-Weighted Average Price):
  Definition: Simple average of prices over time
  Use Case: Illiquid markets, even exposure

Close Price:
  Definition: End-of-day closing price
  Use Case: Index tracking, rebalancing

POV (Percentage of Volume):
  Definition: Execute as fixed % of market volume
  Use Case: Large orders in liquid stocks
```

### Market Impact Model

**Square Root Model** (Industry Standard):
```
Impact = σ × √(Q/V) × κ

Where:
  σ = Daily volatility
  Q = Order quantity
  V = Average daily volume
  κ = Impact coefficient (typically 0.1-0.5)
```

## Implementation Patterns

### 1. TWAP (Time-Weighted Average Price)

```python
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional
import numpy as np

@dataclass
class OrderSlice:
    """Single slice of a parent order"""
    quantity: int
    scheduled_time: datetime
    executed_time: Optional[datetime] = None
    fill_price: Optional[float] = None
    status: str = "pending"  # pending, filled, cancelled

@dataclass
class TWAPOrder:
    """TWAP execution strategy"""
    symbol: str
    total_quantity: int
    side: str  # 'buy' or 'sell'
    start_time: datetime
    end_time: datetime
    n_slices: int
    randomize: bool = True
    variance_pct: float = 0.1  # ±10% randomization

    def __post_init__(self):
        self.slices = self._generate_slices()

    def _generate_slices(self) -> List[OrderSlice]:
        """Generate order slices with optional randomization"""
        duration = (self.end_time - self.start_time).total_seconds()
        interval = duration / self.n_slices

        base_qty = self.total_quantity // self.n_slices
        remainder = self.total_quantity % self.n_slices

        slices = []
        for i in range(self.n_slices):
            # Distribute remainder across first slices
            qty = base_qty + (1 if i < remainder else 0)

            # Calculate scheduled time
            scheduled = self.start_time + timedelta(seconds=interval * i)

            if self.randomize:
                # Add random offset (±variance_pct of interval)
                offset = np.random.uniform(-1, 1) * self.variance_pct * interval
                scheduled += timedelta(seconds=offset)

                # Randomize quantity slightly
                qty_var = int(qty * self.variance_pct * np.random.uniform(-1, 1))
                qty = max(1, qty + qty_var)

            slices.append(OrderSlice(
                quantity=qty,
                scheduled_time=scheduled
            ))

        # Ensure total quantity is exact
        actual_total = sum(s.quantity for s in slices)
        if actual_total != self.total_quantity:
            slices[-1].quantity += (self.total_quantity - actual_total)

        return sorted(slices, key=lambda s: s.scheduled_time)

    def get_next_slice(self, current_time: datetime) -> Optional[OrderSlice]:
        """Get next pending slice ready for execution"""
        for slice in self.slices:
            if slice.status == "pending" and slice.scheduled_time <= current_time:
                return slice
        return None

    def mark_filled(self, slice: OrderSlice, fill_price: float, fill_time: datetime):
        """Mark slice as filled"""
        slice.fill_price = fill_price
        slice.executed_time = fill_time
        slice.status = "filled"

    @property
    def avg_fill_price(self) -> Optional[float]:
        """Calculate average fill price"""
        filled = [s for s in self.slices if s.status == "filled"]
        if not filled:
            return None

        total_value = sum(s.quantity * s.fill_price for s in filled)
        total_qty = sum(s.quantity for s in filled)
        return total_value / total_qty

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled"""
        filled_qty = sum(s.quantity for s in self.slices if s.status == "filled")
        return filled_qty / self.total_quantity


# Usage
def create_twap_order(
    symbol: str,
    quantity: int,
    side: str,
    duration_minutes: int = 60,
    n_slices: int = 12
) -> TWAPOrder:
    """Create TWAP order for execution"""
    now = datetime.now()
    return TWAPOrder(
        symbol=symbol,
        total_quantity=quantity,
        side=side,
        start_time=now,
        end_time=now + timedelta(minutes=duration_minutes),
        n_slices=n_slices,
        randomize=True
    )
```

### 2. VWAP (Volume-Weighted Average Price)

```python
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd
import numpy as np

@dataclass
class VWAPProfile:
    """Historical volume profile for VWAP targeting"""
    time_buckets: List[str]  # e.g., ['09:30', '09:35', ...]
    volume_pcts: List[float]  # Percentage of daily volume per bucket

    @classmethod
    def from_historical(cls, intraday_volume: pd.DataFrame) -> 'VWAPProfile':
        """
        Build volume profile from historical intraday data

        Args:
            intraday_volume: DataFrame with columns [time, volume]
        """
        # Aggregate volume by time bucket
        profile = intraday_volume.groupby('time')['volume'].mean()
        total = profile.sum()

        return cls(
            time_buckets=list(profile.index),
            volume_pcts=list(profile / total)
        )

class VWAPOrder:
    """VWAP execution strategy tracking market volume profile"""

    def __init__(
        self,
        symbol: str,
        total_quantity: int,
        side: str,
        volume_profile: VWAPProfile,
        participation_rate: float = 0.1,  # Max 10% of bucket volume
        min_slice_qty: int = 100
    ):
        self.symbol = symbol
        self.total_quantity = total_quantity
        self.side = side
        self.profile = volume_profile
        self.participation_rate = participation_rate
        self.min_slice_qty = min_slice_qty

        self.slices = self._generate_slices()
        self.fills: List[Dict] = []

    def _generate_slices(self) -> List[Dict]:
        """Generate slices based on volume profile"""
        slices = []

        for bucket, vol_pct in zip(self.profile.time_buckets, self.profile.volume_pcts):
            target_qty = int(self.total_quantity * vol_pct)

            if target_qty >= self.min_slice_qty:
                slices.append({
                    'time_bucket': bucket,
                    'target_quantity': target_qty,
                    'volume_pct': vol_pct,
                    'filled_quantity': 0,
                    'status': 'pending'
                })

        # Redistribute any remainder
        allocated = sum(s['target_quantity'] for s in slices)
        remainder = self.total_quantity - allocated

        if slices and remainder > 0:
            # Add to largest buckets
            slices.sort(key=lambda x: x['volume_pct'], reverse=True)
            for i, s in enumerate(slices):
                if remainder <= 0:
                    break
                add = min(remainder, int(remainder * s['volume_pct'] / sum(
                    x['volume_pct'] for x in slices[i:])))
                s['target_quantity'] += max(1, add)
                remainder -= max(1, add)

        return slices

    def get_target_for_bucket(self, time_bucket: str, market_volume: int) -> int:
        """
        Get target quantity for current time bucket

        Args:
            time_bucket: Current time bucket (e.g., '10:30')
            market_volume: Observed market volume in this bucket

        Returns:
            Target quantity to execute
        """
        for slice in self.slices:
            if slice['time_bucket'] == time_bucket and slice['status'] == 'pending':
                # Limit to participation rate of market volume
                max_qty = int(market_volume * self.participation_rate)
                remaining = slice['target_quantity'] - slice['filled_quantity']
                return min(remaining, max_qty)

        return 0

    def record_fill(self, time_bucket: str, quantity: int, price: float):
        """Record a fill"""
        self.fills.append({
            'time_bucket': time_bucket,
            'quantity': quantity,
            'price': price
        })

        # Update slice
        for slice in self.slices:
            if slice['time_bucket'] == time_bucket:
                slice['filled_quantity'] += quantity
                if slice['filled_quantity'] >= slice['target_quantity']:
                    slice['status'] = 'filled'
                break

    @property
    def avg_fill_price(self) -> Optional[float]:
        """Calculate average fill price"""
        if not self.fills:
            return None

        total_value = sum(f['quantity'] * f['price'] for f in self.fills)
        total_qty = sum(f['quantity'] for f in self.fills)
        return total_value / total_qty

    def calculate_vwap_slippage(self, market_vwap: float) -> float:
        """
        Calculate slippage vs market VWAP

        Returns:
            Slippage in basis points (positive = paid more for buys)
        """
        if not self.fills:
            return 0.0

        avg_fill = self.avg_fill_price
        if self.side == 'buy':
            slippage = (avg_fill - market_vwap) / market_vwap
        else:
            slippage = (market_vwap - avg_fill) / market_vwap

        return slippage * 10000  # Convert to basis points
```

### 3. Iceberg Orders

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class IcebergOrder:
    """
    Iceberg order - shows only visible portion

    Total quantity is hidden; only 'display_quantity' shown to market
    """
    symbol: str
    total_quantity: int
    display_quantity: int  # Visible portion
    side: str
    limit_price: float
    variance_pct: float = 0.2  # Randomize display qty ±20%

    def __post_init__(self):
        self.remaining_quantity = self.total_quantity
        self.filled_quantity = 0
        self.fills: List[Dict] = []
        self._current_display = self._get_display_qty()

    def _get_display_qty(self) -> int:
        """Get randomized display quantity"""
        if self.variance_pct > 0:
            variance = int(self.display_quantity * self.variance_pct)
            qty = self.display_quantity + np.random.randint(-variance, variance + 1)
        else:
            qty = self.display_quantity

        return min(qty, self.remaining_quantity)

    @property
    def current_display_quantity(self) -> int:
        """Currently visible quantity"""
        return min(self._current_display, self.remaining_quantity)

    def fill(self, quantity: int, price: float) -> bool:
        """
        Process a fill

        Returns:
            True if there's more to fill, False if complete
        """
        actual_fill = min(quantity, self.remaining_quantity)

        self.fills.append({
            'quantity': actual_fill,
            'price': price
        })

        self.filled_quantity += actual_fill
        self.remaining_quantity -= actual_fill

        # Refresh display quantity if fully filled
        if actual_fill >= self._current_display:
            self._current_display = self._get_display_qty()

        return self.remaining_quantity > 0

    @property
    def is_complete(self) -> bool:
        return self.remaining_quantity <= 0

    @property
    def avg_fill_price(self) -> Optional[float]:
        if not self.fills:
            return None

        total_value = sum(f['quantity'] * f['price'] for f in self.fills)
        return total_value / self.filled_quantity


class AdaptiveIceberg(IcebergOrder):
    """
    Adaptive iceberg that adjusts display size based on market conditions
    """

    def __init__(
        self,
        symbol: str,
        total_quantity: int,
        base_display: int,
        side: str,
        limit_price: float,
        min_display: int = 100,
        max_display: int = 1000
    ):
        self.base_display = base_display
        self.min_display = min_display
        self.max_display = max_display
        self._market_depth = 1.0  # Multiplier based on market conditions

        super().__init__(
            symbol=symbol,
            total_quantity=total_quantity,
            display_quantity=base_display,
            side=side,
            limit_price=limit_price
        )

    def update_market_conditions(self, bid_ask_spread: float, avg_spread: float):
        """
        Adjust display size based on market conditions

        Wider spread = smaller display (less liquidity)
        Tighter spread = larger display (more liquidity)
        """
        spread_ratio = avg_spread / bid_ask_spread if bid_ask_spread > 0 else 1.0
        self._market_depth = np.clip(spread_ratio, 0.5, 2.0)

        new_display = int(self.base_display * self._market_depth)
        self.display_quantity = np.clip(new_display, self.min_display, self.max_display)
        self._current_display = self._get_display_qty()
```

### 4. POV (Percentage of Volume)

```python
@dataclass
class POVOrder:
    """
    Percentage of Volume order
    Executes as fixed percentage of market volume
    """
    symbol: str
    total_quantity: int
    side: str
    target_pov: float  # Target % of volume (e.g., 0.10 = 10%)
    max_pov: float = 0.25  # Never exceed 25% of volume
    min_order_size: int = 100

    def __post_init__(self):
        self.filled_quantity = 0
        self.market_volume_traded = 0
        self.fills: List[Dict] = []

    @property
    def remaining_quantity(self) -> int:
        return self.total_quantity - self.filled_quantity

    def calculate_order_size(self, interval_volume: int) -> int:
        """
        Calculate order size based on observed market volume

        Args:
            interval_volume: Market volume in current interval

        Returns:
            Target order size for this interval
        """
        if self.remaining_quantity <= 0:
            return 0

        # Target quantity based on POV
        target = int(interval_volume * self.target_pov)

        # Cap at max POV
        max_qty = int(interval_volume * self.max_pov)
        target = min(target, max_qty)

        # Don't exceed remaining
        target = min(target, self.remaining_quantity)

        # Minimum order size
        if target < self.min_order_size and target < self.remaining_quantity:
            return 0  # Wait for more volume

        return target

    def record_fill(self, quantity: int, price: float, interval_volume: int):
        """Record fill and update tracking"""
        self.fills.append({
            'quantity': quantity,
            'price': price,
            'interval_volume': interval_volume
        })

        self.filled_quantity += quantity
        self.market_volume_traded += interval_volume

    @property
    def actual_pov(self) -> float:
        """Calculate actual participation rate"""
        if self.market_volume_traded == 0:
            return 0.0
        return self.filled_quantity / self.market_volume_traded
```

### 5. Market Impact Model

```python
@dataclass
class MarketImpactModel:
    """
    Square-root market impact model

    Estimates expected price impact and execution cost
    """
    volatility: float  # Daily volatility (e.g., 0.02 = 2%)
    adv: int  # Average daily volume
    bid_ask_spread: float  # In price terms
    impact_coefficient: float = 0.3  # κ, typically 0.1-0.5

    def estimate_temporary_impact(self, quantity: int) -> float:
        """
        Estimate temporary market impact (reverts after execution)

        Returns impact as percentage of price
        """
        participation = quantity / self.adv
        impact = self.volatility * np.sqrt(participation) * self.impact_coefficient
        return impact

    def estimate_permanent_impact(self, quantity: int) -> float:
        """
        Estimate permanent market impact (doesn't revert)

        Typically ~30-50% of temporary impact
        """
        temp_impact = self.estimate_temporary_impact(quantity)
        return temp_impact * 0.4  # 40% permanent

    def estimate_total_cost(
        self,
        quantity: int,
        price: float,
        urgency: float = 0.5  # 0=passive, 1=aggressive
    ) -> Dict[str, float]:
        """
        Estimate total execution cost

        Args:
            quantity: Order quantity
            price: Current price
            urgency: Execution urgency (affects spread cost)

        Returns:
            Dictionary with cost breakdown
        """
        # Spread cost (half spread for each side)
        spread_cost = (self.bid_ask_spread / price) * 0.5 * (1 + urgency)

        # Market impact
        temp_impact = self.estimate_temporary_impact(quantity)
        perm_impact = self.estimate_permanent_impact(quantity)

        # Timing risk (opportunity cost of slow execution)
        timing_risk = self.volatility * np.sqrt(1 - urgency) * 0.5

        total_cost = spread_cost + temp_impact + timing_risk

        return {
            'spread_cost_bps': spread_cost * 10000,
            'temporary_impact_bps': temp_impact * 10000,
            'permanent_impact_bps': perm_impact * 10000,
            'timing_risk_bps': timing_risk * 10000,
            'total_cost_bps': total_cost * 10000,
            'total_cost_dollars': total_cost * price * quantity
        }

    def optimal_execution_horizon(self, quantity: int) -> float:
        """
        Estimate optimal execution time horizon (in days)

        Balances market impact vs timing risk
        """
        participation = quantity / self.adv

        # Almgren-Chriss optimal horizon approximation
        # T* ∝ (Q/V)^(2/3) / σ^(2/3)
        optimal_days = (participation ** (2/3)) / (self.volatility ** (2/3))

        return np.clip(optimal_days, 0.1, 5.0)  # Between 0.1 and 5 days
```

## Production Execution Engine

```python
from dataclasses import dataclass
from typing import Literal, Optional, Dict, List
from datetime import datetime, timedelta
from enum import Enum
import numpy as np

class ExecutionAlgo(Enum):
    TWAP = "twap"
    VWAP = "vwap"
    POV = "pov"
    ICEBERG = "iceberg"
    AGGRESSIVE = "aggressive"

@dataclass
class ExecutionConfig:
    """Configuration for execution algorithm"""
    algo: ExecutionAlgo
    duration_minutes: int = 60
    participation_rate: float = 0.1
    display_qty: Optional[int] = None  # For iceberg
    limit_price: Optional[float] = None
    urgency: float = 0.5  # 0=passive, 1=aggressive

class ExecutionEngine:
    """
    Production execution engine with multiple algorithms

    Manages order lifecycle, tracks execution quality
    """

    def __init__(self, impact_model: MarketImpactModel):
        self.impact_model = impact_model
        self.active_orders: Dict[str, any] = {}
        self.completed_orders: List[Dict] = []

    def create_order(
        self,
        order_id: str,
        symbol: str,
        quantity: int,
        side: Literal['buy', 'sell'],
        config: ExecutionConfig,
        arrival_price: float
    ) -> Dict:
        """Create and register new execution order"""

        # Pre-trade cost estimate
        cost_estimate = self.impact_model.estimate_total_cost(
            quantity, arrival_price, config.urgency
        )

        if config.algo == ExecutionAlgo.TWAP:
            order = TWAPOrder(
                symbol=symbol,
                total_quantity=quantity,
                side=side,
                start_time=datetime.now(),
                end_time=datetime.now() + timedelta(minutes=config.duration_minutes),
                n_slices=max(1, config.duration_minutes // 5)  # Slice every 5 min
            )
        elif config.algo == ExecutionAlgo.ICEBERG:
            display = config.display_qty or (quantity // 10)
            order = IcebergOrder(
                symbol=symbol,
                total_quantity=quantity,
                display_quantity=display,
                side=side,
                limit_price=config.limit_price or arrival_price
            )
        elif config.algo == ExecutionAlgo.POV:
            order = POVOrder(
                symbol=symbol,
                total_quantity=quantity,
                side=side,
                target_pov=config.participation_rate
            )
        else:
            raise ValueError(f"Unsupported algo: {config.algo}")

        self.active_orders[order_id] = {
            'order': order,
            'config': config,
            'arrival_price': arrival_price,
            'cost_estimate': cost_estimate,
            'start_time': datetime.now()
        }

        return {
            'order_id': order_id,
            'estimated_cost_bps': cost_estimate['total_cost_bps'],
            'estimated_cost_dollars': cost_estimate['total_cost_dollars']
        }

    def get_execution_metrics(self, order_id: str) -> Dict:
        """Get execution quality metrics for order"""
        if order_id not in self.active_orders:
            return {}

        order_data = self.active_orders[order_id]
        order = order_data['order']
        arrival = order_data['arrival_price']

        avg_fill = order.avg_fill_price
        if avg_fill is None:
            return {'status': 'no fills yet'}

        # Implementation shortfall (vs arrival price)
        if order.side == 'buy':
            impl_shortfall = (avg_fill - arrival) / arrival
        else:
            impl_shortfall = (arrival - avg_fill) / arrival

        return {
            'order_id': order_id,
            'filled_quantity': order.filled_quantity if hasattr(order, 'filled_quantity')
                              else sum(s.quantity for s in order.slices if s.status == 'filled'),
            'remaining_quantity': order.remaining_quantity if hasattr(order, 'remaining_quantity')
                                 else order.total_quantity - sum(s.quantity for s in order.slices if s.status == 'filled'),
            'avg_fill_price': avg_fill,
            'arrival_price': arrival,
            'implementation_shortfall_bps': impl_shortfall * 10000,
            'estimated_cost_bps': order_data['cost_estimate']['total_cost_bps'],
            'fill_rate': order.fill_rate if hasattr(order, 'fill_rate') else None
        }


# Usage Example
if __name__ == "__main__":
    # Create impact model
    impact = MarketImpactModel(
        volatility=0.02,  # 2% daily vol
        adv=1_000_000,    # 1M shares ADV
        bid_ask_spread=0.02,  # $0.02 spread
        impact_coefficient=0.3
    )

    # Estimate costs for 50,000 share order
    costs = impact.estimate_total_cost(
        quantity=50_000,
        price=100.00,
        urgency=0.5
    )

    print("Pre-Trade Cost Estimate:")
    print(f"  Spread Cost: {costs['spread_cost_bps']:.1f} bps")
    print(f"  Market Impact: {costs['temporary_impact_bps']:.1f} bps")
    print(f"  Timing Risk: {costs['timing_risk_bps']:.1f} bps")
    print(f"  Total: {costs['total_cost_bps']:.1f} bps (${costs['total_cost_dollars']:.2f})")

    # Create TWAP order
    engine = ExecutionEngine(impact)
    order_info = engine.create_order(
        order_id="ORD-001",
        symbol="AAPL",
        quantity=50_000,
        side='buy',
        config=ExecutionConfig(
            algo=ExecutionAlgo.TWAP,
            duration_minutes=60
        ),
        arrival_price=100.00
    )

    print(f"\nOrder Created: {order_info['order_id']}")
    print(f"Estimated Cost: {order_info['estimated_cost_bps']:.1f} bps")
```

## Best Practices

1. **Match algorithm to order characteristics**:
   - Small orders: Market/limit orders
   - Large liquid: VWAP/TWAP
   - Large illiquid: Iceberg, POV
   - Urgent: Aggressive/arrival price

2. **Monitor execution quality in real-time**:
   - Track vs benchmark (VWAP, arrival)
   - Alert on excessive slippage
   - Adapt parameters mid-execution

3. **Use pre-trade analytics**:
   - Estimate market impact before execution
   - Set realistic cost expectations
   - Choose appropriate urgency level

4. **Randomize to avoid detection**:
   - Vary slice sizes ±10-20%
   - Randomize timing within intervals
   - Use adaptive display quantities

## Common Pitfalls

❌ **Using same display size** for all icebergs
✅ Randomize and adapt to market conditions

❌ **Ignoring market impact** for large orders
✅ Use impact models, reduce participation rate

❌ **VWAP without volume profile** data
✅ Build accurate historical profiles, update regularly

❌ **Fixed execution horizon** regardless of urgency
✅ Adjust based on alpha decay, risk tolerance

❌ **No post-trade analysis**
✅ Compare actual vs estimated costs, refine models

## Quality Standards

- **Fill Rate**: >95% within execution window
- **Slippage vs VWAP**: <5 bps for liquid names
- **Implementation Shortfall**: Within 2x pre-trade estimate
- **Detection Avoidance**: Pattern recognition tests pass
- **Latency**: <50ms order slice generation

---

**Skill Type**: Finance - Order Execution
**Complexity**: Complex
**Typical Usage**: Activated when algorithmic-trading-engineer needs execution algorithms
**Performance**: Real-time slice generation with <10ms latency
