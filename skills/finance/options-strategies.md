---
name: options-strategies
description: Load when user needs options strategies, spreads, straddles, iron condors, covered calls, or multi-leg options positions. Covers payoff diagrams, breakeven calculations, and strategy selection.
trigger_keywords: [options strategy, spread, straddle, strangle, iron condor, iron butterfly, covered call, protective put, collar, vertical spread, calendar spread, diagonal spread, bull spread, bear spread, credit spread, debit spread, butterfly spread, options payoff, max profit, max loss, breakeven]
---

# Options Strategies Skill

Comprehensive guide to multi-leg options strategies with payoff analysis, Greeks profiles, and selection criteria.

## Strategy Categories

| Category | Risk Profile | Market View | Examples |
|----------|--------------|-------------|----------|
| **Directional** | Defined/Undefined | Bullish/Bearish | Long call, covered call |
| **Vertical Spreads** | Defined | Directional | Bull call, bear put |
| **Neutral/Volatility** | Defined | Range-bound/Volatile | Iron condor, straddle |
| **Calendar** | Defined | Time decay | Horizontal spread |

## Directional Strategies

### Long Call
**View**: Bullish | **Risk**: Limited to premium | **Reward**: Unlimited

```python
def long_call_payoff(S: float, K: float, premium: float) -> float:
    """Payoff at expiration for long call"""
    return max(S - K, 0) - premium

# Example: Buy 100 call for $3.00
# Max Loss: $300 (premium paid)
# Max Profit: Unlimited
# Breakeven: K + premium = $103
```

### Long Put
**View**: Bearish | **Risk**: Limited to premium | **Reward**: K - premium

```python
def long_put_payoff(S: float, K: float, premium: float) -> float:
    """Payoff at expiration for long put"""
    return max(K - S, 0) - premium

# Breakeven: K - premium
```

### Covered Call
**View**: Neutral to slightly bullish | **Risk**: Stock downside | **Reward**: Premium + (K - stock cost)

```python
def covered_call_payoff(S: float, K: float, stock_cost: float, premium: float) -> float:
    """
    Long 100 shares + Short 1 call
    """
    stock_pnl = S - stock_cost
    call_pnl = premium - max(S - K, 0)
    return stock_pnl + call_pnl

# Max Profit: premium + (K - stock_cost) if S >= K
# Max Loss: stock_cost - premium if S -> 0
# Breakeven: stock_cost - premium
```

### Protective Put (Married Put)
**View**: Bullish with downside protection | **Risk**: Limited | **Reward**: Unlimited - premium

```python
def protective_put_payoff(S: float, K: float, stock_cost: float, premium: float) -> float:
    """
    Long 100 shares + Long 1 put
    """
    stock_pnl = S - stock_cost
    put_pnl = max(K - S, 0) - premium
    return stock_pnl + put_pnl

# Max Loss: stock_cost + premium - K
# Max Profit: Unlimited (minus premium paid)
```

## Vertical Spreads

### Bull Call Spread (Debit)
**View**: Moderately bullish | **Risk**: Net debit | **Reward**: Width - debit

```python
def bull_call_spread(
    S: float,
    K_long: float,
    K_short: float,
    premium_long: float,
    premium_short: float
) -> dict:
    """
    Buy lower strike call, Sell higher strike call
    Same expiration
    """
    net_debit = premium_long - premium_short
    width = K_short - K_long

    if S <= K_long:
        payoff = -net_debit
    elif S >= K_short:
        payoff = width - net_debit
    else:
        payoff = (S - K_long) - net_debit

    return {
        'payoff': payoff,
        'max_profit': width - net_debit,
        'max_loss': net_debit,
        'breakeven': K_long + net_debit
    }

# Example: Buy 100 call @ $5, Sell 105 call @ $2
# Net Debit: $3, Max Profit: $2, Breakeven: $103
```

### Bear Put Spread (Debit)
**View**: Moderately bearish | **Risk**: Net debit | **Reward**: Width - debit

```python
def bear_put_spread(
    S: float,
    K_long: float,  # Higher strike (buy)
    K_short: float, # Lower strike (sell)
    premium_long: float,
    premium_short: float
) -> dict:
    """
    Buy higher strike put, Sell lower strike put
    """
    net_debit = premium_long - premium_short
    width = K_long - K_short

    if S >= K_long:
        payoff = -net_debit
    elif S <= K_short:
        payoff = width - net_debit
    else:
        payoff = (K_long - S) - net_debit

    return {
        'payoff': payoff,
        'max_profit': width - net_debit,
        'max_loss': net_debit,
        'breakeven': K_long - net_debit
    }
```

### Bull Put Spread (Credit)
**View**: Neutral to bullish | **Risk**: Width - credit | **Reward**: Net credit

```python
def bull_put_spread(
    S: float,
    K_short: float,  # Higher strike (sell)
    K_long: float,   # Lower strike (buy)
    premium_short: float,
    premium_long: float
) -> dict:
    """
    Sell higher strike put, Buy lower strike put
    Credit received upfront
    """
    net_credit = premium_short - premium_long
    width = K_short - K_long

    if S >= K_short:
        payoff = net_credit  # Keep full credit
    elif S <= K_long:
        payoff = net_credit - width  # Max loss
    else:
        payoff = net_credit - (K_short - S)

    return {
        'payoff': payoff,
        'max_profit': net_credit,
        'max_loss': width - net_credit,
        'breakeven': K_short - net_credit
    }
```

### Bear Call Spread (Credit)
**View**: Neutral to bearish | **Risk**: Width - credit | **Reward**: Net credit

```python
def bear_call_spread(
    S: float,
    K_short: float,  # Lower strike (sell)
    K_long: float,   # Higher strike (buy)
    premium_short: float,
    premium_long: float
) -> dict:
    """
    Sell lower strike call, Buy higher strike call
    """
    net_credit = premium_short - premium_long
    width = K_long - K_short

    if S <= K_short:
        payoff = net_credit
    elif S >= K_long:
        payoff = net_credit - width
    else:
        payoff = net_credit - (S - K_short)

    return {
        'payoff': payoff,
        'max_profit': net_credit,
        'max_loss': width - net_credit,
        'breakeven': K_short + net_credit
    }
```

## Neutral/Volatility Strategies

### Long Straddle
**View**: High volatility expected | **Risk**: Total premium | **Reward**: Unlimited

```python
def long_straddle(
    S: float,
    K: float,
    call_premium: float,
    put_premium: float
) -> dict:
    """
    Buy ATM call + Buy ATM put (same strike, same expiration)
    Profit from big moves in either direction
    """
    total_premium = call_premium + put_premium
    call_payoff = max(S - K, 0)
    put_payoff = max(K - S, 0)
    payoff = call_payoff + put_payoff - total_premium

    return {
        'payoff': payoff,
        'max_profit': float('inf'),  # Unlimited
        'max_loss': total_premium,
        'breakeven_upper': K + total_premium,
        'breakeven_lower': K - total_premium
    }

# When to use: Before earnings, FDA announcements, major events
# Greeks: Long gamma, long vega, negative theta
```

### Long Strangle
**View**: High volatility expected | **Risk**: Total premium | **Reward**: Unlimited

```python
def long_strangle(
    S: float,
    K_call: float,  # OTM call strike (higher)
    K_put: float,   # OTM put strike (lower)
    call_premium: float,
    put_premium: float
) -> dict:
    """
    Buy OTM call + Buy OTM put
    Cheaper than straddle but needs bigger move
    """
    total_premium = call_premium + put_premium
    call_payoff = max(S - K_call, 0)
    put_payoff = max(K_put - S, 0)
    payoff = call_payoff + put_payoff - total_premium

    return {
        'payoff': payoff,
        'max_loss': total_premium,
        'breakeven_upper': K_call + total_premium,
        'breakeven_lower': K_put - total_premium
    }
```

### Iron Condor
**View**: Low volatility, range-bound | **Risk**: Width - credit | **Reward**: Net credit

```python
def iron_condor(
    S: float,
    K_put_long: float,   # Lowest strike
    K_put_short: float,  # Lower middle
    K_call_short: float, # Upper middle
    K_call_long: float,  # Highest strike
    net_credit: float
) -> dict:
    """
    Bull put spread + Bear call spread
    Profit if stock stays between short strikes

    Structure:
    - Buy OTM put (K_put_long)
    - Sell OTM put (K_put_short)
    - Sell OTM call (K_call_short)
    - Buy OTM call (K_call_long)
    """
    put_width = K_put_short - K_put_long
    call_width = K_call_long - K_call_short
    max_width = max(put_width, call_width)

    if K_put_short <= S <= K_call_short:
        payoff = net_credit  # Max profit zone
    elif S < K_put_long:
        payoff = net_credit - put_width
    elif S > K_call_long:
        payoff = net_credit - call_width
    elif S < K_put_short:
        payoff = net_credit - (K_put_short - S)
    else:  # S > K_call_short
        payoff = net_credit - (S - K_call_short)

    return {
        'payoff': payoff,
        'max_profit': net_credit,
        'max_loss': max_width - net_credit,
        'breakeven_lower': K_put_short - net_credit,
        'breakeven_upper': K_call_short + net_credit,
        'profit_zone': (K_put_short, K_call_short)
    }

# Example: Stock at $100
# Buy 90 put, Sell 95 put, Sell 105 call, Buy 110 call
# Net credit: $2.00
# Max profit: $200 if stock between $95-$105
# Max loss: $300 (width $5 - credit $2)
```

### Iron Butterfly
**View**: Minimal movement expected | **Risk**: Width - credit | **Reward**: Net credit

```python
def iron_butterfly(
    S: float,
    K_put_long: float,   # Lower wing
    K_middle: float,     # ATM (both short options)
    K_call_long: float,  # Upper wing
    net_credit: float
) -> dict:
    """
    Sell ATM straddle + Buy OTM strangle for protection
    Tighter profit zone than iron condor, higher credit

    Structure:
    - Buy OTM put
    - Sell ATM put
    - Sell ATM call
    - Buy OTM call
    """
    width = K_middle - K_put_long  # Assuming symmetric

    if S == K_middle:
        payoff = net_credit  # Max profit at exactly ATM
    elif S <= K_put_long:
        payoff = net_credit - width
    elif S >= K_call_long:
        payoff = net_credit - width
    elif S < K_middle:
        payoff = net_credit - (K_middle - S)
    else:
        payoff = net_credit - (S - K_middle)

    return {
        'payoff': payoff,
        'max_profit': net_credit,
        'max_loss': width - net_credit,
        'breakeven_lower': K_middle - net_credit,
        'breakeven_upper': K_middle + net_credit
    }
```

### Butterfly Spread
**View**: Stock pinned near strike | **Risk**: Net debit | **Reward**: Width - debit

```python
def long_call_butterfly(
    S: float,
    K_lower: float,
    K_middle: float,
    K_upper: float,
    net_debit: float
) -> dict:
    """
    Buy 1 lower call, Sell 2 middle calls, Buy 1 upper call
    Max profit if stock at middle strike at expiration
    """
    width = K_middle - K_lower

    if S <= K_lower:
        payoff = -net_debit
    elif S >= K_upper:
        payoff = -net_debit
    elif S <= K_middle:
        payoff = (S - K_lower) - net_debit
    else:
        payoff = (K_upper - S) - net_debit

    return {
        'payoff': payoff,
        'max_profit': width - net_debit,
        'max_loss': net_debit,
        'breakeven_lower': K_lower + net_debit,
        'breakeven_upper': K_upper - net_debit
    }
```

## Calendar Spreads

### Long Calendar Spread (Horizontal)
**View**: Low near-term volatility, stable price | **Risk**: Net debit | **Reward**: Variable

```python
def calendar_spread_concept():
    """
    Sell near-term option, Buy longer-term option (same strike)
    Profit from time decay differential

    Structure:
    - Sell front-month call/put at strike K
    - Buy back-month call/put at same strike K

    Greeks Profile:
    - Positive theta (front month decays faster)
    - Positive vega (benefits from IV increase)
    - Near-zero delta (if ATM)

    Max Profit: At expiration of front month, stock at K
    Max Loss: Net debit paid
    """
    pass

# Best when: IV is low and expected to rise
# Risk: Stock moves significantly away from strike
```

### Diagonal Spread
**View**: Directional + time decay | **Risk**: Net debit | **Reward**: Variable

```python
def diagonal_spread_concept():
    """
    Calendar spread with different strikes

    Bullish Diagonal (Poor Man's Covered Call):
    - Buy deep ITM LEAPS call (delta ~0.80)
    - Sell near-term OTM call

    Synthetic covered call with less capital
    """
    pass
```

## Strategy Selection Framework

### By Market View

```python
STRATEGY_BY_VIEW = {
    'strong_bullish': ['long_call', 'bull_call_spread'],
    'moderate_bullish': ['bull_put_spread', 'covered_call'],
    'neutral': ['iron_condor', 'iron_butterfly', 'short_straddle'],
    'moderate_bearish': ['bear_call_spread', 'bear_put_spread'],
    'strong_bearish': ['long_put', 'bear_put_spread'],
    'high_volatility': ['long_straddle', 'long_strangle'],
    'low_volatility': ['iron_condor', 'short_straddle', 'calendar_spread']
}
```

### By Risk Tolerance

```python
STRATEGY_BY_RISK = {
    'defined_risk': [
        'vertical_spreads',
        'iron_condor',
        'iron_butterfly',
        'butterfly'
    ],
    'undefined_risk': [
        'naked_call',  # Unlimited risk - avoid!
        'naked_put',   # Risk to zero
        'short_straddle'
    ],
    'limited_risk_unlimited_reward': [
        'long_call',
        'long_put',
        'long_straddle',
        'long_strangle'
    ]
}
```

### Greeks Profile by Strategy

| Strategy | Delta | Gamma | Theta | Vega |
|----------|-------|-------|-------|------|
| Long Call | + | + | - | + |
| Covered Call | + (reduced) | - | + | - |
| Bull Call Spread | + | +/- | -/+ | +/- |
| Iron Condor | ~0 | - | + | - |
| Long Straddle | ~0 | + | - | + |
| Calendar Spread | ~0 | - | + | + |

## Production Implementation

```python
from dataclasses import dataclass
from typing import Literal, List
from enum import Enum

class StrategyType(Enum):
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    IRON_CONDOR = "iron_condor"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

@dataclass
class OptionLeg:
    strike: float
    expiration: str  # ISO date
    option_type: Literal['call', 'put']
    action: Literal['buy', 'sell']
    quantity: int
    premium: float

@dataclass
class StrategyAnalysis:
    max_profit: float
    max_loss: float
    breakeven: List[float]
    profit_probability: float  # From delta approximation
    risk_reward_ratio: float
    net_debit_credit: float

class OptionsStrategyAnalyzer:
    """Analyze and construct options strategies"""

    def analyze_strategy(self, legs: List[OptionLeg]) -> StrategyAnalysis:
        """Calculate strategy metrics from legs"""
        net_premium = sum(
            leg.premium * leg.quantity * (1 if leg.action == 'sell' else -1)
            for leg in legs
        )

        # Calculate payoff at various prices
        # ... implementation

        return StrategyAnalysis(
            max_profit=self._calc_max_profit(legs),
            max_loss=self._calc_max_loss(legs),
            breakeven=self._calc_breakevens(legs),
            profit_probability=self._estimate_pop(legs),
            risk_reward_ratio=abs(self._calc_max_profit(legs) / self._calc_max_loss(legs)),
            net_debit_credit=net_premium
        )

    def build_iron_condor(
        self,
        underlying_price: float,
        width: float = 5.0,
        wing_distance: float = 10.0
    ) -> List[OptionLeg]:
        """Auto-construct iron condor around current price"""
        put_short = underlying_price - wing_distance
        put_long = put_short - width
        call_short = underlying_price + wing_distance
        call_long = call_short + width

        return [
            OptionLeg(put_long, "2024-01-19", "put", "buy", 1, 0),
            OptionLeg(put_short, "2024-01-19", "put", "sell", 1, 0),
            OptionLeg(call_short, "2024-01-19", "call", "sell", 1, 0),
            OptionLeg(call_long, "2024-01-19", "call", "buy", 1, 0),
        ]
```

## Best Practices

1. **Always define max loss** before entering any trade
2. **Use defined-risk strategies** for beginners (spreads, not naked options)
3. **Size positions** based on max loss, not premium
4. **Consider assignment risk** for short options near expiration
5. **Monitor Greeks** especially as expiration approaches (gamma risk)
6. **Close winners early** (50-75% of max profit) to reduce risk

## Common Pitfalls

- **Pin risk**: Short options at exactly the strike at expiration
- **Early assignment**: American-style options on dividend-paying stocks
- **Liquidity**: Wide bid-ask spreads on multi-leg orders
- **Gamma explosion**: Near-expiration ATM options have extreme gamma
- **IV crush**: Post-earnings volatility collapse destroys long vega positions

---

**Skill Type**: Finance - Options Strategies
**Complexity**: Intermediate to Advanced
**Typical Usage**: Activated when building multi-leg options positions
