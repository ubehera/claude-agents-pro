# Tier 08: Finance & Trading Agents

Production-ready Claude Code agents for algorithmic trading, quantitative analysis, and systematic trading strategies. Specialized for stocks and options trading across multiple brokers (Alpaca, E*TRADE, Fidelity).

## Agent Catalog

| Agent | Focus | Primary Use Case | Tools |
|-------|-------|------------------|-------|
| **market-data-engineer** | Data pipelines | Real-time & historical market data infrastructure | Read, Write, MultiEdit, Bash, WebFetch, Task |
| **quantitative-analyst** | Technical & statistical analysis | Indicators, Greeks, statistical models | Read, Write, MultiEdit, Bash, Task, WebSearch |
| **trading-strategy-architect** | Strategy development | Backtesting frameworks, walk-forward analysis | Read, Write, MultiEdit, Bash, Task, WebSearch |
| **trading-risk-manager** | Risk management | Position sizing, portfolio optimization, VaR | Read, Write, MultiEdit, Bash, Task |
| **algorithmic-trading-engineer** | Order execution | Multi-broker integration, order management | Read, Write, MultiEdit, Bash, WebFetch, Task |
| **equity-research-analyst** | Fundamental analysis | DCF valuation, financial statement analysis | Read, Write, MultiEdit, WebSearch, Task |
| **trading-ml-specialist** | ML for trading | Walk-forward ML, trading-specific features | Read, Write, MultiEdit, Bash, Task, WebSearch |
| **trading-compliance-officer** | Regulatory compliance | PDT rules, wash sales, trade reporting | Read, Write, MultiEdit, WebSearch, Task |
| **portfolio-manager** | Portfolio construction | Multi-strategy allocation, rebalancing, performance attribution | Read, Write, MultiEdit, Bash, Task |
| **finance-glossary** | Domain reference | Canonical terminology, domain boundaries, ubiquitous language | Read, Grep, Glob |

## Multi-Agent Workflows

### Complete Strategy Development
```
User: "Develop a momentum trading strategy for tech stocks"

Workflow:
1. market-data-engineer: Set up NASDAQ-100 data pipeline
2. quantitative-analyst: Calculate momentum indicators (ROC, ADX, MACD)
3. trading-strategy-architect: Backtest with vectorbt, walk-forward validation
4. trading-risk-manager: Apply Kelly criterion position sizing
5. algorithmic-trading-engineer: Deploy to paper trading (Alpaca)
6. trading-compliance-officer: Validate PDT compliance

Result: Validated, risk-managed strategy ready for paper trading
```

### Fundamental + Technical Strategy
```
User: "Find undervalued stocks with strong momentum"

Workflow:
1. equity-research-analyst: Screen for low P/E, high ROE stocks
2. quantitative-analyst: Filter for positive momentum (RSI, MACD)
3. trading-strategy-architect: Backtest combined strategy
4. trading-risk-manager: Portfolio optimization across selected stocks
5. algorithmic-trading-engineer: Execute trades with TWAP algorithm

Result: Fundamental stock selection with technical entry timing
```

### ML-Enhanced Trading
```
User: "Use machine learning to predict price movements"

Workflow:
1. market-data-engineer: Fetch training data (5 years historical)
2. quantitative-analyst: Generate technical features (50+ indicators)
3. trading-ml-specialist: Train XGBoost with walk-forward validation
4. trading-strategy-architect: Integrate ML signals into backtest
5. trading-risk-manager: Validate risk metrics (Sharpe >1.5)
6. algorithmic-trading-engineer: Deploy if validated

Result: ML-powered strategy with realistic performance expectations
```

## Agent Coordination Patterns

### Data Flow
```
market-data-engineer → quantitative-analyst → trading-strategy-architect
                                            → equity-research-analyst
```

### Validation Chain
```
trading-strategy-architect → trading-risk-manager → algorithmic-trading-engineer
                                                  → trading-compliance-officer
```

### Advanced Workflows
```
equity-research-analyst (stock selection)
    ↓
quantitative-analyst (technical timing)
    ↓
trading-ml-specialist (ML enhancement)
    ↓
trading-strategy-architect (backtest)
    ↓
trading-risk-manager (position sizing)
    ↓
algorithmic-trading-engineer (execution)
    ↓
trading-compliance-officer (audit)
```

## Installation & Setup

```bash
# Install all finance agents
cd claude-agents-pro
./scripts/install-agents.sh --user

# Restart Claude Code to load agents
# Verify installation
./scripts/verify-agents.sh

# Test specific agent
# User: "Set up real-time stock data pipeline"
# Expected: market-data-engineer agent selected
```

## Technology Stack

### Data & Infrastructure
- **Databases**: TimescaleDB, QuestDB, PostgreSQL, Redis
- **Message Queues**: Kafka, Redis Streams
- **Data Providers**: Alpaca, Polygon.io, IEX Cloud, Alpha Vantage

### Analysis & ML
- **Python Libraries**: NumPy, pandas, scipy, scikit-learn, XGBoost
- **Technical Analysis**: TA-Lib, pandas-ta
- **Options**: Black-Scholes, Greeks calculation
- **Backtesting**: vectorbt, backtrader, zipline

### Execution & Risk
- **Brokers**: Alpaca API, E*TRADE OAuth, Fidelity
- **Portfolio Optimization**: cvxpy, scipy.optimize, PyPortfolioOpt
- **Risk Metrics**: VaR, CVaR, Sharpe, Sortino, max drawdown

## Quality Standards

### Data Quality
- Uptime: >99.95% during market hours
- Latency: <100ms from exchange to database
- Completeness: >99.99% (no missing bars)

### Strategy Quality
- Sharpe Ratio: >1.5 out-of-sample
- Max Drawdown: <20%
- Walk-Forward: >60% positive windows

### Execution Quality
- Order Success: >99.9%
- Slippage: <0.05% vs expected
- Fill Rate: >95% (limit orders)

### Compliance
- PDT Accuracy: 100%
- Wash Sale Detection: 100%
- Audit Trail: Complete for all trades

## Key Differentiators

1. **Production-Ready**: Complete code examples with error handling, logging, type hints
2. **Multi-Broker**: Abstraction layer supports Alpaca, E*TRADE, Fidelity
3. **Options Support**: Greeks, options chains, options strategies
4. **Risk-First**: Mandatory risk validation before any trade execution
5. **Compliance Built-In**: PDT rules, wash sales automated
6. **Realistic Backtesting**: Transaction costs, slippage, walk-forward analysis

## Common Pitfalls Avoided

- ✅ Walk-forward validation (not just in-sample optimization)
- ✅ Transaction costs included in all backtests
- ✅ No look-ahead bias in technical indicators
- ✅ Position sizing mandatory (no unlimited risk)
- ✅ Risk limits enforced before execution
- ✅ Regulatory compliance (PDT, wash sales) automated

## Getting Started

### Beginner: Paper Trading
```
1. Set up Alpaca paper trading account (free)
2. Use market-data-engineer to fetch historical data
3. Use quantitative-analyst to calculate indicators
4. Use trading-strategy-architect to backtest
5. Deploy to Alpaca paper trading
```

### Intermediate: Live Trading (Small Account)
```
1. Connect real broker (Alpaca, E*TRADE)
2. Start with small position sizes (<5% per trade)
3. Enable trading-compliance-officer for PDT monitoring
4. Use trading-risk-manager for position sizing
5. Monitor execution quality daily
```

### Advanced: Multiple Strategies
```
1. Develop 3-5 uncorrelated strategies
2. Use trading-risk-manager for portfolio optimization
3. Deploy with algorithmic-trading-engineer
4. Monitor with trading-ml-specialist for model decay
5. Monthly performance review
```

## Support & Maintenance

- **Agent Updates**: Quarterly reviews for broker API changes
- **Regulatory Updates**: Annual review of compliance rules
- **Technology Updates**: Python library updates (pandas, NumPy, vectorbt)
- **Broker Support**: Add new brokers as requested

## Next Steps

1. **Install agents**: `./scripts/install-agents.sh --user`
2. **Test routing**: Ask "Set up market data pipeline" → should invoke market-data-engineer
3. **Explore workflows**: Try the multi-agent workflows above
4. **Start building**: Begin with paper trading, validate strategies thoroughly

---

**Disclaimer**: These agents provide tools and frameworks for trading. They do not provide financial advice. Past performance does not guarantee future results. Trading involves substantial risk of loss. Consult with qualified financial advisors before trading.

---
Licensed under Apache-2.0.
