---
name: futures-strategies
description: Load when user needs futures trading strategies, contango, backwardation, roll strategies, basis trading, or futures hedging. Covers contract mechanics, term structure, and spread strategies.
trigger_keywords: [futures, futures contract, contango, backwardation, roll, basis, margin, settlement, expiration, calendar spread, inter-commodity spread, hedge ratio, mark-to-market, futures curve, term structure, roll yield, cost of carry]
---

# Futures Strategies Skill

Comprehensive guide to futures contract mechanics, term structure analysis, and trading strategies.

## Futures Contract Mechanics

### Contract Specifications

```python
from dataclasses import dataclass
from datetime import date
from typing import Literal

@dataclass
class FuturesContract:
    """Standard futures contract specification"""
    symbol: str              # e.g., "ES", "CL", "GC"
    underlying: str          # e.g., "S&P 500", "Crude Oil", "Gold"
    contract_size: float     # Multiplier (e.g., 50 for ES, 1000 for CL)
    tick_size: float         # Minimum price movement
    tick_value: float        # Dollar value per tick
    expiration: date
    settlement: Literal['cash', 'physical']
    margin_initial: float    # Initial margin requirement
    margin_maintenance: float # Maintenance margin

# Common contracts
ES = FuturesContract(
    symbol="ES",
    underlying="S&P 500 E-mini",
    contract_size=50,
    tick_size=0.25,
    tick_value=12.50,  # 50 * 0.25
    expiration=date(2024, 3, 15),
    settlement='cash',
    margin_initial=12_000,
    margin_maintenance=10_800
)

CL = FuturesContract(
    symbol="CL",
    underlying="Crude Oil",
    contract_size=1000,  # barrels
    tick_size=0.01,
    tick_value=10.00,  # 1000 * 0.01
    expiration=date(2024, 2, 20),
    settlement='physical',
    margin_initial=6_500,
    margin_maintenance=5_900
)
```

### Margin & Mark-to-Market

```python
@dataclass
class FuturesPosition:
    contract: FuturesContract
    quantity: int  # Positive = long, negative = short
    entry_price: float
    current_price: float

    @property
    def notional_value(self) -> float:
        """Total contract value"""
        return abs(self.quantity) * self.contract.contract_size * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        """Mark-to-market P&L"""
        price_change = self.current_price - self.entry_price
        return self.quantity * self.contract.contract_size * price_change

    @property
    def margin_required(self) -> float:
        """Initial margin for position"""
        return abs(self.quantity) * self.contract.margin_initial

    def margin_call_price(self) -> float:
        """Price at which margin call triggers"""
        margin_buffer = self.contract.margin_initial - self.contract.margin_maintenance
        price_move = margin_buffer / (abs(self.quantity) * self.contract.contract_size)
        if self.quantity > 0:  # Long
            return self.entry_price - price_move
        else:  # Short
            return self.entry_price + price_move

# Example: Long 2 ES at 4500
position = FuturesPosition(
    contract=ES,
    quantity=2,
    entry_price=4500,
    current_price=4520
)
print(f"Notional: ${position.notional_value:,.0f}")  # $452,000
print(f"P&L: ${position.unrealized_pnl:,.0f}")       # $2,000 (20 pts * 50 * 2)
print(f"Margin: ${position.margin_required:,.0f}")   # $24,000
print(f"Leverage: {position.notional_value / position.margin_required:.1f}x")  # ~19x
```

## Term Structure

### Contango vs Backwardation

```python
from enum import Enum
from typing import List, Tuple

class TermStructure(Enum):
    CONTANGO = "contango"           # Far months > Near months
    BACKWARDATION = "backwardation" # Near months > Far months
    FLAT = "flat"

def analyze_term_structure(
    futures_curve: List[Tuple[date, float]]  # [(expiration, price), ...]
) -> dict:
    """
    Analyze futures term structure

    Contango: F > S (futures premium)
    - Normal for storable commodities (storage costs)
    - Negative roll yield for long positions

    Backwardation: F < S (futures discount)
    - Supply shortages, convenience yield
    - Positive roll yield for long positions
    """
    sorted_curve = sorted(futures_curve, key=lambda x: x[0])

    # Calculate front-to-back spread
    front_price = sorted_curve[0][1]
    back_price = sorted_curve[-1][1]
    spread_pct = (back_price - front_price) / front_price * 100

    # Determine structure
    if spread_pct > 0.5:
        structure = TermStructure.CONTANGO
    elif spread_pct < -0.5:
        structure = TermStructure.BACKWARDATION
    else:
        structure = TermStructure.FLAT

    # Calculate annualized roll yield
    days_to_back = (sorted_curve[-1][0] - sorted_curve[0][0]).days
    annualized_spread = spread_pct * (365 / days_to_back)

    return {
        'structure': structure,
        'front_price': front_price,
        'back_price': back_price,
        'spread_pct': spread_pct,
        'annualized_roll_yield': -annualized_spread,  # Negative for contango longs
        'curve': sorted_curve
    }

# Example: Oil futures curve
oil_curve = [
    (date(2024, 2, 20), 75.50),  # Feb
    (date(2024, 3, 20), 76.20),  # Mar
    (date(2024, 4, 22), 76.80),  # Apr
    (date(2024, 6, 20), 77.50),  # Jun
]
analysis = analyze_term_structure(oil_curve)
# Structure: CONTANGO, Roll yield: negative for longs
```

### Cost of Carry Model

```python
def theoretical_futures_price(
    spot_price: float,
    risk_free_rate: float,
    storage_cost: float,      # Annual storage cost as % of spot
    convenience_yield: float, # Benefit of holding physical
    time_to_expiry: float     # Years
) -> float:
    """
    Cost of Carry model for futures pricing

    F = S * e^((r + s - y) * T)

    Where:
    - F = Futures price
    - S = Spot price
    - r = Risk-free rate
    - s = Storage cost
    - y = Convenience yield
    - T = Time to expiry
    """
    import math
    cost_of_carry = risk_free_rate + storage_cost - convenience_yield
    return spot_price * math.exp(cost_of_carry * time_to_expiry)

# Example: Gold futures
gold_spot = 2000
gold_futures = theoretical_futures_price(
    spot_price=gold_spot,
    risk_free_rate=0.05,      # 5%
    storage_cost=0.005,       # 0.5% annual storage
    convenience_yield=0.0,    # No convenience yield for gold
    time_to_expiry=0.25       # 3 months
)
print(f"Theoretical 3-month gold futures: ${gold_futures:.2f}")
# ~$2027.70 (contango due to carry costs)
```

## Roll Strategies

### Roll Mechanics

```python
from datetime import timedelta

def calculate_roll_cost(
    front_price: float,
    back_price: float,
    contracts: int,
    contract_size: float
) -> dict:
    """
    Calculate cost of rolling futures position

    Roll = Sell front month, Buy back month (for longs)
    """
    roll_spread = back_price - front_price
    roll_cost = roll_spread * contracts * contract_size

    return {
        'front_price': front_price,
        'back_price': back_price,
        'roll_spread': roll_spread,
        'roll_spread_pct': roll_spread / front_price * 100,
        'total_roll_cost': roll_cost,
        'structure': 'contango' if roll_spread > 0 else 'backwardation'
    }

# Example: Rolling 10 CL contracts
roll = calculate_roll_cost(
    front_price=75.50,
    back_price=76.20,
    contracts=10,
    contract_size=1000
)
print(f"Roll cost: ${roll['total_roll_cost']:,.0f}")  # $7,000
print(f"Roll spread: {roll['roll_spread_pct']:.2f}%") # 0.93%
```

### Roll Timing Strategies

```python
class RollStrategy(Enum):
    CALENDAR = "calendar"        # Roll on specific dates
    VOLUME = "volume"            # Roll when back month volume exceeds front
    OPEN_INTEREST = "oi"         # Roll based on open interest shift
    SPREAD = "spread"            # Roll when spread narrows

def optimal_roll_timing(
    front_volume: int,
    back_volume: int,
    front_oi: int,
    back_oi: int,
    days_to_expiry: int,
    roll_spread: float
) -> dict:
    """
    Determine optimal roll timing based on multiple factors
    """
    signals = {}

    # Volume crossover (back > front suggests roll)
    signals['volume_signal'] = back_volume > front_volume

    # Open interest shift
    oi_ratio = back_oi / (front_oi + 1)
    signals['oi_signal'] = oi_ratio > 0.5

    # Calendar (roll 5-10 days before expiry)
    signals['calendar_signal'] = days_to_expiry <= 10

    # Spread narrowing opportunity
    # (track historical spread and roll when below average)

    roll_urgency = sum([
        signals['volume_signal'],
        signals['oi_signal'],
        signals['calendar_signal']
    ])

    return {
        'signals': signals,
        'roll_urgency': roll_urgency,  # 0-3 scale
        'recommendation': 'roll_now' if roll_urgency >= 2 else 'wait'
    }
```

## Spread Strategies

### Calendar Spread (Intra-Commodity)

```python
def calendar_spread(
    front_price: float,
    back_price: float,
    contract_size: float,
    view: Literal['bull', 'bear']
) -> dict:
    """
    Calendar Spread: Long one expiry, Short another (same commodity)

    Bull Calendar: Buy near, Sell far
    - Profit if spread widens (backwardation increases)
    - Used when expecting near-term supply tightness

    Bear Calendar: Sell near, Buy far
    - Profit if spread narrows (contango increases)
    - Used when expecting near-term supply glut
    """
    spread = front_price - back_price

    if view == 'bull':
        # Buy front, sell back - profit if spread widens
        position = {'front': 'long', 'back': 'short'}
        pnl_per_point = contract_size  # Profit as spread widens
    else:
        # Sell front, buy back - profit if spread narrows
        position = {'front': 'short', 'back': 'long'}
        pnl_per_point = -contract_size

    return {
        'spread': spread,
        'position': position,
        'margin': contract_size * abs(spread) * 0.1,  # Spread margin ~10%
        'breakeven': spread,
        'max_loss': 'unlimited but reduced vs outright'
    }

# Example: CL calendar spread
cl_calendar = calendar_spread(
    front_price=75.50,
    back_price=76.20,
    contract_size=1000,
    view='bull'
)
# Current spread: -$0.70 (contango)
# Bull view: expect spread to become less negative or positive
```

### Inter-Commodity Spread

```python
def crack_spread(
    crude_price: float,
    gasoline_price: float,
    heating_oil_price: float
) -> dict:
    """
    3-2-1 Crack Spread: Refinery margin proxy

    Buy 3 crude, Sell 2 gasoline + 1 heating oil
    Represents refinery economics
    """
    # Convert to per-barrel basis
    # Gasoline: 42 gallons/barrel, quoted in $/gallon
    gasoline_per_barrel = gasoline_price * 42
    heating_per_barrel = heating_oil_price * 42

    crack_spread = (2 * gasoline_per_barrel + heating_per_barrel) / 3 - crude_price

    return {
        'crude': crude_price,
        'gasoline_barrel': gasoline_per_barrel,
        'heating_barrel': heating_per_barrel,
        'crack_spread': crack_spread,
        'margin_per_barrel': crack_spread
    }

def gold_silver_ratio(gold_price: float, silver_price: float) -> dict:
    """
    Gold/Silver Ratio spread

    Ratio > 80: Silver undervalued, buy silver/sell gold
    Ratio < 60: Gold undervalued, buy gold/sell silver
    """
    ratio = gold_price / silver_price

    if ratio > 80:
        signal = 'long_silver_short_gold'
    elif ratio < 60:
        signal = 'long_gold_short_silver'
    else:
        signal = 'neutral'

    return {
        'ratio': ratio,
        'signal': signal,
        'historical_mean': 70  # Approximate
    }
```

### Basis Trading

```python
def basis_trade(
    spot_price: float,
    futures_price: float,
    days_to_expiry: int
) -> dict:
    """
    Basis = Spot - Futures

    Basis Trade: Exploit mispricing between spot and futures

    Long Basis: Buy spot, Sell futures
    - Profit if basis strengthens (becomes less negative/more positive)
    - Convergence trade as expiry approaches

    Short Basis: Sell spot, Buy futures
    - Profit if basis weakens
    """
    basis = spot_price - futures_price
    basis_pct = basis / spot_price * 100
    annualized_basis = basis_pct * (365 / days_to_expiry)

    return {
        'spot': spot_price,
        'futures': futures_price,
        'basis': basis,
        'basis_pct': basis_pct,
        'annualized_basis': annualized_basis,
        'days_to_expiry': days_to_expiry,
        'convergence_expected': True  # Basis -> 0 at expiry
    }

# Example: Arbitrage opportunity
basis = basis_trade(
    spot_price=2000,
    futures_price=2030,
    days_to_expiry=90
)
# Basis: -$30 (-1.5%)
# If carry cost is < 1.5% quarterly, long basis trade is profitable
```

## Hedging with Futures

### Hedge Ratio Calculation

```python
import numpy as np

def minimum_variance_hedge_ratio(
    spot_returns: np.ndarray,
    futures_returns: np.ndarray
) -> dict:
    """
    Optimal hedge ratio minimizing portfolio variance

    h* = Cov(S, F) / Var(F) = ρ * (σS / σF)
    """
    covariance = np.cov(spot_returns, futures_returns)[0, 1]
    futures_variance = np.var(futures_returns)
    correlation = np.corrcoef(spot_returns, futures_returns)[0, 1]

    hedge_ratio = covariance / futures_variance

    # Hedge effectiveness (R²)
    hedge_effectiveness = correlation ** 2

    return {
        'hedge_ratio': hedge_ratio,
        'correlation': correlation,
        'hedge_effectiveness': hedge_effectiveness,
        'interpretation': f"{hedge_effectiveness*100:.1f}% of variance eliminated"
    }

def calculate_hedge_contracts(
    exposure_value: float,
    hedge_ratio: float,
    futures_price: float,
    contract_size: float
) -> int:
    """
    Calculate number of futures contracts for hedge
    """
    hedge_notional = exposure_value * hedge_ratio
    contracts = hedge_notional / (futures_price * contract_size)
    return round(contracts)

# Example: Hedge $10M equity portfolio
contracts_needed = calculate_hedge_contracts(
    exposure_value=10_000_000,
    hedge_ratio=0.95,  # Portfolio beta to S&P
    futures_price=4500,
    contract_size=50    # ES multiplier
)
print(f"Contracts needed: {contracts_needed}")  # ~42 ES contracts
```

### Long vs Short Hedge

```python
def long_hedge(
    expected_purchase_price: float,
    futures_entry: float,
    futures_exit: float,
    actual_spot: float,
    quantity: float
) -> dict:
    """
    Long Hedge: Protect against price INCREASE

    Used by: Processors, manufacturers, importers
    Action: Buy futures now, sell at purchase time
    """
    # Futures P&L
    futures_pnl = (futures_exit - futures_entry) * quantity

    # Effective purchase price
    effective_price = actual_spot - (futures_pnl / quantity)

    return {
        'spot_price_paid': actual_spot,
        'futures_pnl': futures_pnl,
        'effective_price': effective_price,
        'hedge_benefit': expected_purchase_price - effective_price
    }

def short_hedge(
    expected_sale_price: float,
    futures_entry: float,
    futures_exit: float,
    actual_spot: float,
    quantity: float
) -> dict:
    """
    Short Hedge: Protect against price DECREASE

    Used by: Producers, farmers, miners
    Action: Sell futures now, buy back at sale time
    """
    # Futures P&L (short position)
    futures_pnl = (futures_entry - futures_exit) * quantity

    # Effective sale price
    effective_price = actual_spot + (futures_pnl / quantity)

    return {
        'spot_price_received': actual_spot,
        'futures_pnl': futures_pnl,
        'effective_price': effective_price,
        'hedge_benefit': effective_price - expected_sale_price
    }
```

## Risk Management

### Position Sizing

```python
def futures_position_size(
    account_equity: float,
    risk_per_trade: float,     # As decimal (e.g., 0.02 = 2%)
    stop_loss_points: float,
    contract: FuturesContract
) -> dict:
    """
    Calculate position size based on risk tolerance
    """
    dollar_risk = account_equity * risk_per_trade
    risk_per_contract = stop_loss_points * contract.contract_size

    max_contracts = int(dollar_risk / risk_per_contract)

    # Also check margin constraint
    margin_based_max = int(account_equity * 0.5 / contract.margin_initial)

    contracts = min(max_contracts, margin_based_max)

    return {
        'contracts': contracts,
        'dollar_risk': contracts * risk_per_contract,
        'risk_pct': (contracts * risk_per_contract) / account_equity * 100,
        'margin_used': contracts * contract.margin_initial,
        'margin_pct': (contracts * contract.margin_initial) / account_equity * 100
    }

# Example: $100K account, 2% risk, 20 point stop on ES
position = futures_position_size(
    account_equity=100_000,
    risk_per_trade=0.02,
    stop_loss_points=20,
    contract=ES
)
# Max 2 contracts (2 * 20 * 50 = $2000 risk = 2%)
```

### Margin Call Management

```python
def monitor_margin(
    equity: float,
    positions: List[FuturesPosition],
    margin_buffer: float = 0.25  # 25% buffer above maintenance
) -> dict:
    """
    Monitor margin and warn of potential margin calls
    """
    total_initial = sum(p.margin_required for p in positions)
    total_maintenance = sum(
        abs(p.quantity) * p.contract.margin_maintenance
        for p in positions
    )
    total_pnl = sum(p.unrealized_pnl for p in positions)

    current_equity = equity + total_pnl
    excess_margin = current_equity - total_maintenance
    margin_cushion = excess_margin / total_maintenance if total_maintenance > 0 else float('inf')

    warning_level = 'safe'
    if margin_cushion < margin_buffer:
        warning_level = 'warning'
    if margin_cushion < 0:
        warning_level = 'margin_call'

    return {
        'current_equity': current_equity,
        'maintenance_required': total_maintenance,
        'excess_margin': excess_margin,
        'margin_cushion_pct': margin_cushion * 100,
        'warning_level': warning_level,
        'action': 'reduce_position' if warning_level != 'safe' else 'none'
    }
```

## Strategy Selection

### By Market Condition

```python
STRATEGY_BY_CONDITION = {
    'strong_trend_up': ['long_outright', 'long_breakout'],
    'strong_trend_down': ['short_outright', 'short_breakout'],
    'range_bound': ['calendar_spread', 'mean_reversion'],
    'high_volatility': ['reduce_size', 'wider_stops'],
    'contango_steep': ['short_front_long_back', 'avoid_long_roll'],
    'backwardation': ['long_front', 'roll_yield_capture'],
    'convergence_play': ['basis_trade', 'cash_and_carry']
}
```

### Risk/Reward Comparison

| Strategy | Risk | Reward | Margin | Complexity |
|----------|------|--------|--------|------------|
| Outright Long/Short | High | Unlimited | Full | Low |
| Calendar Spread | Medium | Limited | Reduced | Medium |
| Inter-Commodity | Medium | Limited | Reduced | High |
| Basis Trade | Low | Limited | Varies | High |
| Hedged Position | Defined | Defined | Full | Medium |

## Best Practices

1. **Understand leverage**: Futures provide 10-20x leverage; size accordingly
2. **Monitor margin daily**: Don't get caught by margin calls
3. **Plan rolls in advance**: Roll before liquidity dries up in front month
4. **Use spread orders**: Execute calendar spreads as single order for better fills
5. **Track term structure**: Contango/backwardation affects long-term returns
6. **Match hedge horizon**: Align futures expiry with hedging need

## Common Pitfalls

- **Ignoring roll costs**: Contango can erode returns for long-term longs
- **Over-leveraging**: Full margin usage leaves no room for adverse moves
- **Rolling too late**: Low liquidity near expiry means poor execution
- **Basis risk**: Futures may not perfectly track spot/cash position
- **Gap risk**: Futures can gap significantly on weekend news

---

**Skill Type**: Finance - Futures Trading
**Complexity**: Intermediate to Advanced
**Typical Usage**: Activated when building futures positions or hedging
