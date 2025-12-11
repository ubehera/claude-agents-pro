---
name: quantconnect
description: Load when user needs QuantConnect/Lean engine development, algorithm framework, universe selection, alpha models, or Lean API integration. Covers QCAlgorithm patterns and cloud deployment.
trigger_keywords: [quantconnect, lean engine, qcalgorithm, AddEquity, universe selection, alpha model, portfolio construction model, lean api, quantconnect cloud, coarse universe, fine universe, scheduled events]
---

# QuantConnect / Lean Engine Skill

Comprehensive QuantConnect algorithm development using the Lean engine framework.

## QCAlgorithm Basics

### Algorithm Structure

```python
from AlgorithmImports import *

class MyStrategy(QCAlgorithm):
    """
    All QuantConnect algorithms inherit from QCAlgorithm
    """

    def Initialize(self):
        """
        Called once at start - configure algorithm here
        """
        # Time range for backtest
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)

        # Starting capital
        self.SetCash(100000)

        # Brokerage model (affects fees, margin, etc.)
        self.SetBrokerageModel(BrokerageName.InteractiveBrokersBrokerage)

        # Add securities
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.aapl = self.AddEquity("AAPL", Resolution.Minute).Symbol

        # Benchmark for comparison
        self.SetBenchmark("SPY")

        # Warm up indicators
        self.SetWarmUp(200, Resolution.Daily)

    def OnData(self, data: Slice):
        """
        Called on each data event (bar or tick)
        """
        if self.IsWarmingUp:
            return

        # Check if we have data
        if not data.ContainsKey(self.spy):
            return

        # Access price data
        price = data[self.spy].Close

        # Place orders
        if not self.Portfolio[self.spy].Invested:
            self.SetHoldings(self.spy, 0.5)  # 50% allocation

    def OnOrderEvent(self, orderEvent: OrderEvent):
        """Called when order status changes"""
        if orderEvent.Status == OrderStatus.Filled:
            self.Debug(f"Filled: {orderEvent.Symbol} @ {orderEvent.FillPrice}")
```

### Data Subscriptions

```python
def Initialize(self):
    # Equities
    self.AddEquity("SPY", Resolution.Minute)
    self.AddEquity("AAPL", Resolution.Second)  # Tick data

    # Options
    option = self.AddOption("SPY", Resolution.Minute)
    option.SetFilter(-5, 5, 0, 30)  # Strike range, days to expiry

    # Futures
    future = self.AddFuture(Futures.Indices.SP500EMini, Resolution.Minute)
    future.SetFilter(0, 90)  # Days to expiry range

    # Forex
    self.AddForex("EURUSD", Resolution.Hour)

    # Crypto
    self.AddCrypto("BTCUSD", Resolution.Daily)

    # Custom data
    self.AddData(MyCustomData, "CUSTOM", Resolution.Daily)

class MyCustomData(PythonData):
    """Custom data source"""

    def GetSource(self, config, date, isLive):
        return SubscriptionDataSource(
            f"https://api.example.com/data/{date.strftime('%Y%m%d')}.csv",
            SubscriptionTransportMedium.RemoteFile
        )

    def Reader(self, config, line, date, isLive):
        if not line.strip():
            return None
        data = MyCustomData()
        data.Symbol = config.Symbol
        parts = line.split(',')
        data.Time = datetime.strptime(parts[0], "%Y-%m-%d")
        data.Value = float(parts[1])
        return data
```

## Universe Selection

### Coarse/Fine Universe Selection

```python
class UniverseSelectionAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Universe selection - runs daily by default
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        self.UniverseSettings.Resolution = Resolution.Daily
        self.UniverseSettings.Leverage = 2

    def CoarseSelectionFunction(self, coarse: List[CoarseFundamental]) -> List[Symbol]:
        """
        First filter - price/volume based (fast)
        Runs on all ~8000 US equities
        """
        # Filter by price and volume
        filtered = [x for x in coarse
                   if x.Price > 10
                   and x.DollarVolume > 10_000_000
                   and x.HasFundamentalData]

        # Sort by dollar volume, take top 100
        sorted_by_volume = sorted(filtered,
                                  key=lambda x: x.DollarVolume,
                                  reverse=True)[:100]

        return [x.Symbol for x in sorted_by_volume]

    def FineSelectionFunction(self, fine: List[FineFundamental]) -> List[Symbol]:
        """
        Second filter - fundamental data (slower, more detailed)
        Only runs on symbols from coarse filter
        """
        # Filter by market cap and PE ratio
        filtered = [x for x in fine
                   if x.MarketCap > 1_000_000_000  # $1B+
                   and x.ValuationRatios.PERatio > 0
                   and x.ValuationRatios.PERatio < 30]

        # Sort by market cap, take top 20
        sorted_by_cap = sorted(filtered,
                              key=lambda x: x.MarketCap,
                              reverse=True)[:20]

        return [x.Symbol for x in sorted_by_cap]

    def OnSecuritiesChanged(self, changes: SecurityChanges):
        """Called when universe changes"""
        # Liquidate removed securities
        for security in changes.RemovedSecurities:
            if security.Invested:
                self.Liquidate(security.Symbol)

        # Equal weight new securities
        for security in changes.AddedSecurities:
            self.SetHoldings(security.Symbol, 1.0 / 20)
```

### ETF Constituents Universe

```python
def Initialize(self):
    # Track S&P 500 constituents via SPY
    self.AddUniverse(self.Universe.ETF("SPY", self.UniverseSettings, self.ETFFilter))

def ETFFilter(self, constituents: List[ETFConstituentData]) -> List[Symbol]:
    """Filter ETF constituents"""
    # Get top 50 by weight
    sorted_by_weight = sorted(constituents,
                              key=lambda x: x.Weight,
                              reverse=True)[:50]
    return [x.Symbol for x in sorted_by_weight]
```

## Alpha Models

### Framework-Based Algorithm

```python
from AlgorithmImports import *

class FrameworkAlgorithm(QCAlgorithm):
    """
    Using the Algorithm Framework for modular design
    """

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        # Universe Selection Model
        self.SetUniverseSelection(
            CoarseFundamentalUniverseSelectionModel(self.CoarseFilter)
        )

        # Alpha Model - generates trading signals
        self.SetAlpha(MomentumAlphaModel())

        # Portfolio Construction - converts signals to targets
        self.SetPortfolioConstruction(
            EqualWeightingPortfolioConstructionModel()
        )

        # Risk Management
        self.SetRiskManagement(
            MaximumDrawdownPercentPerSecurity(0.05)  # 5% max loss per position
        )

        # Execution Model
        self.SetExecution(
            ImmediateExecutionModel()
        )

    def CoarseFilter(self, coarse):
        return [x.Symbol for x in coarse
                if x.Price > 10 and x.DollarVolume > 5_000_000][:50]


class MomentumAlphaModel(AlphaModel):
    """
    Custom alpha model - momentum based
    """

    def __init__(self, lookback=12, holding_period=21):
        self.lookback = lookback
        self.holding_period = holding_period
        self.securities = {}

    def Update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        """
        Generate insights (trading signals)
        """
        insights = []

        for symbol, security_data in self.securities.items():
            if not data.ContainsKey(symbol):
                continue

            # Calculate momentum
            history = algorithm.History(symbol, self.lookback * 21, Resolution.Daily)
            if len(history) < self.lookback * 21:
                continue

            returns = history['close'].pct_change(self.lookback * 21).iloc[-1]

            # Generate insight
            if returns > 0.1:  # 10%+ momentum
                insights.append(Insight.Price(
                    symbol,
                    timedelta(days=self.holding_period),
                    InsightDirection.Up,
                    magnitude=returns,
                    confidence=0.7
                ))
            elif returns < -0.1:
                insights.append(Insight.Price(
                    symbol,
                    timedelta(days=self.holding_period),
                    InsightDirection.Down,
                    magnitude=abs(returns),
                    confidence=0.7
                ))

        return insights

    def OnSecuritiesChanged(self, algorithm: QCAlgorithm, changes: SecurityChanges):
        """Track universe changes"""
        for security in changes.AddedSecurities:
            self.securities[security.Symbol] = {}

        for security in changes.RemovedSecurities:
            self.securities.pop(security.Symbol, None)
```

### Portfolio Construction Models

```python
class RiskParityPortfolioConstruction(PortfolioConstructionModel):
    """
    Risk parity - equal risk contribution from each asset
    """

    def __init__(self, rebalance_period=30):
        self.rebalance_period = rebalance_period
        self.last_rebalance = None

    def CreateTargets(self, algorithm: QCAlgorithm,
                      insights: List[Insight]) -> List[PortfolioTarget]:
        """Convert insights to portfolio targets"""

        # Check rebalance timing
        if self.last_rebalance and \
           (algorithm.Time - self.last_rebalance).days < self.rebalance_period:
            return []

        self.last_rebalance = algorithm.Time

        # Get active symbols from insights
        symbols = [i.Symbol for i in insights if i.Direction != InsightDirection.Flat]

        if not symbols:
            return []

        # Calculate volatilities
        volatilities = {}
        for symbol in symbols:
            history = algorithm.History(symbol, 60, Resolution.Daily)
            if len(history) > 20:
                volatilities[symbol] = history['close'].pct_change().std()

        if not volatilities:
            return []

        # Inverse volatility weighting (simple risk parity)
        total_inv_vol = sum(1/v for v in volatilities.values())
        weights = {s: (1/v) / total_inv_vol for s, v in volatilities.items()}

        return [PortfolioTarget(s, w) for s, w in weights.items()]
```

## Scheduled Events

```python
def Initialize(self):
    self.SetStartDate(2020, 1, 1)
    self.SetCash(100000)

    self.spy = self.AddEquity("SPY", Resolution.Minute).Symbol

    # Schedule rebalance at market open every Monday
    self.Schedule.On(
        self.DateRules.Every(DayOfWeek.Monday),
        self.TimeRules.AfterMarketOpen(self.spy, 30),  # 30 min after open
        self.Rebalance
    )

    # Schedule risk check every day at close
    self.Schedule.On(
        self.DateRules.EveryDay(self.spy),
        self.TimeRules.BeforeMarketClose(self.spy, 10),  # 10 min before close
        self.RiskCheck
    )

    # Monthly rebalance on first trading day
    self.Schedule.On(
        self.DateRules.MonthStart(self.spy),
        self.TimeRules.At(10, 0),
        self.MonthlyRebalance
    )

def Rebalance(self):
    """Weekly rebalance logic"""
    self.SetHoldings(self.spy, 0.6)

def RiskCheck(self):
    """Daily risk check"""
    if self.Portfolio.TotalUnrealizedProfit < -5000:
        self.Liquidate()

def MonthlyRebalance(self):
    """Monthly rebalance"""
    self.Log(f"Monthly rebalance: {self.Time}")
```

## Indicators

```python
def Initialize(self):
    self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

    # Built-in indicators (auto-updated)
    self.sma = self.SMA(self.spy, 20, Resolution.Daily)
    self.ema = self.EMA(self.spy, 20, Resolution.Daily)
    self.rsi = self.RSI(self.spy, 14, MovingAverageType.Wilders, Resolution.Daily)
    self.macd = self.MACD(self.spy, 12, 26, 9, MovingAverageType.Exponential, Resolution.Daily)
    self.bb = self.BB(self.spy, 20, 2, Resolution.Daily)
    self.atr = self.ATR(self.spy, 14, Resolution.Daily)

    # Warm up indicators with historical data
    self.SetWarmUp(200, Resolution.Daily)

def OnData(self, data):
    if self.IsWarmingUp:
        return

    # Check indicator ready
    if not self.sma.IsReady or not self.rsi.IsReady:
        return

    price = data[self.spy].Close

    # Use indicator values
    if price > self.sma.Current.Value and self.rsi.Current.Value < 30:
        self.SetHoldings(self.spy, 1.0)
    elif price < self.sma.Current.Value and self.rsi.Current.Value > 70:
        self.Liquidate()

    # MACD signal
    if self.macd.Current.Value > self.macd.Signal.Current.Value:
        self.Debug("MACD bullish crossover")

    # Bollinger Bands
    if price < self.bb.LowerBand.Current.Value:
        self.Debug("Price below lower band")
```

## Options Trading

```python
class OptionsAlgorithm(QCAlgorithm):

    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)

        equity = self.AddEquity("SPY", Resolution.Minute)
        option = self.AddOption("SPY", Resolution.Minute)

        # Filter options chain
        option.SetFilter(
            minStrike=-10,  # 10 strikes below ATM
            maxStrike=10,   # 10 strikes above ATM
            minExpiry=timedelta(days=7),
            maxExpiry=timedelta(days=45)
        )

        self.symbol = option.Symbol

    def OnData(self, data):
        if not self.Portfolio.Invested:
            # Get options chain
            chain = data.OptionChains.get(self.symbol)
            if chain is None:
                return

            # Filter for ATM calls expiring in ~30 days
            underlying_price = self.Securities["SPY"].Price

            calls = [x for x in chain
                    if x.Right == OptionRight.Call
                    and abs(x.Strike - underlying_price) < 5
                    and 20 < (x.Expiry - self.Time).days < 40]

            if not calls:
                return

            # Select nearest to ATM
            contract = min(calls, key=lambda x: abs(x.Strike - underlying_price))

            # Buy the call
            self.MarketOrder(contract.Symbol, 1)

    def OnOrderEvent(self, orderEvent):
        self.Debug(f"Order: {orderEvent}")
```

## Lean API Integration

### API Client Setup

```python
import requests
from typing import Dict, List, Optional

class LeanAPIClient:
    """
    QuantConnect Lean API client for cloud operations
    """

    def __init__(self, user_id: str, api_token: str):
        """
        Get credentials from quantconnect.com/account
        """
        self.base_url = "https://www.quantconnect.com/api/v2"
        self.auth = (user_id, api_token)

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make authenticated API request"""
        url = f"{self.base_url}/{endpoint}"
        response = requests.request(method, url, auth=self.auth, **kwargs)
        response.raise_for_status()
        return response.json()

    # ==================== Projects ====================

    def list_projects(self) -> List[Dict]:
        """List all projects"""
        result = self._request("GET", "projects/read")
        return result.get('projects', [])

    def create_project(self, name: str, language: str = "Py") -> Dict:
        """Create new project (Py or C#)"""
        return self._request("POST", "projects/create",
                            data={"name": name, "language": language})

    def get_project(self, project_id: int) -> Dict:
        """Get project details"""
        return self._request("GET", f"projects/read",
                            params={"projectId": project_id})

    def delete_project(self, project_id: int) -> Dict:
        """Delete a project"""
        return self._request("POST", "projects/delete",
                            data={"projectId": project_id})

    # ==================== Files ====================

    def list_files(self, project_id: int) -> List[Dict]:
        """List files in project"""
        result = self._request("GET", "files/read",
                              params={"projectId": project_id})
        return result.get('files', [])

    def read_file(self, project_id: int, filename: str) -> str:
        """Read file content"""
        result = self._request("GET", "files/read",
                              params={"projectId": project_id, "name": filename})
        return result.get('files', [{}])[0].get('content', '')

    def update_file(self, project_id: int, filename: str, content: str) -> Dict:
        """Update/create file"""
        return self._request("POST", "files/update",
                            data={
                                "projectId": project_id,
                                "name": filename,
                                "content": content
                            })

    # ==================== Backtests ====================

    def create_backtest(self, project_id: int, name: str,
                       compile_id: str = None) -> Dict:
        """
        Launch a backtest

        Returns backtest ID for tracking
        """
        data = {"projectId": project_id, "backtestName": name}
        if compile_id:
            data["compileId"] = compile_id
        return self._request("POST", "backtests/create", data=data)

    def read_backtest(self, project_id: int, backtest_id: str) -> Dict:
        """
        Get backtest results

        Returns full backtest data including:
        - Statistics (sharpe, returns, drawdown)
        - Charts data
        - Orders/trades
        - Holdings
        """
        return self._request("GET", "backtests/read",
                            params={
                                "projectId": project_id,
                                "backtestId": backtest_id
                            })

    def list_backtests(self, project_id: int) -> List[Dict]:
        """List all backtests for project"""
        result = self._request("GET", "backtests/read",
                              params={"projectId": project_id})
        return result.get('backtests', [])

    def delete_backtest(self, project_id: int, backtest_id: str) -> Dict:
        """Delete a backtest"""
        return self._request("POST", "backtests/delete",
                            data={
                                "projectId": project_id,
                                "backtestId": backtest_id
                            })

    # ==================== Live Trading ====================

    def create_live(self, project_id: int, compile_id: str,
                   node_id: str, brokerage: Dict) -> Dict:
        """
        Deploy live algorithm

        brokerage: Dict with brokerage-specific credentials
        """
        return self._request("POST", "live/create",
                            json={
                                "projectId": project_id,
                                "compileId": compile_id,
                                "nodeId": node_id,
                                **brokerage
                            })

    def read_live(self, project_id: int, deploy_id: str) -> Dict:
        """Get live algorithm status and results"""
        return self._request("GET", "live/read",
                            params={
                                "projectId": project_id,
                                "deployId": deploy_id
                            })

    def stop_live(self, project_id: int) -> Dict:
        """Stop live algorithm"""
        return self._request("POST", "live/stop",
                            data={"projectId": project_id})

    def liquidate_live(self, project_id: int) -> Dict:
        """Liquidate all positions and stop"""
        return self._request("POST", "live/liquidate",
                            data={"projectId": project_id})

    # ==================== Compile ====================

    def compile_project(self, project_id: int) -> Dict:
        """
        Compile project before backtest/live

        Returns compile ID needed for backtest/live
        """
        return self._request("POST", "compile/create",
                            data={"projectId": project_id})

    def read_compile(self, project_id: int, compile_id: str) -> Dict:
        """Check compile status"""
        return self._request("GET", "compile/read",
                            params={
                                "projectId": project_id,
                                "compileId": compile_id
                            })


# ==================== Usage Examples ====================

def run_backtest_workflow():
    """Complete workflow: compile → backtest → get results"""

    client = LeanAPIClient(
        user_id="YOUR_USER_ID",
        api_token="YOUR_API_TOKEN"
    )

    project_id = 12345678

    # 1. Compile the project
    compile_result = client.compile_project(project_id)
    compile_id = compile_result['compileId']
    print(f"Compile started: {compile_id}")

    # 2. Wait for compile (poll status)
    import time
    while True:
        status = client.read_compile(project_id, compile_id)
        if status['state'] == 'BuildSuccess':
            print("Compile successful!")
            break
        elif status['state'] == 'BuildError':
            print(f"Compile failed: {status.get('logs')}")
            return
        time.sleep(2)

    # 3. Create backtest
    backtest_result = client.create_backtest(
        project_id=project_id,
        name=f"Backtest_{time.strftime('%Y%m%d_%H%M%S')}",
        compile_id=compile_id
    )
    backtest_id = backtest_result['backtestId']
    print(f"Backtest started: {backtest_id}")

    # 4. Poll for completion
    while True:
        bt = client.read_backtest(project_id, backtest_id)
        if bt.get('completed', False):
            break
        print(f"Progress: {bt.get('progress', 0)}%")
        time.sleep(5)

    # 5. Extract results
    results = client.read_backtest(project_id, backtest_id)
    stats = results.get('result', {}).get('Statistics', {})

    print("\n=== Backtest Results ===")
    print(f"Total Return: {stats.get('Total Net Profit', 'N/A')}")
    print(f"Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}")
    print(f"Max Drawdown: {stats.get('Drawdown', 'N/A')}")
    print(f"Win Rate: {stats.get('Win Rate', 'N/A')}")

    return results


def extract_backtest_data(results: Dict) -> Dict:
    """
    Parse backtest results into structured data
    """
    result = results.get('result', {})

    return {
        'statistics': result.get('Statistics', {}),
        'runtime_statistics': result.get('RuntimeStatistics', {}),
        'profit_loss': result.get('ProfitLoss', {}),
        'total_orders': result.get('TotalOrders', 0),
        'charts': {
            name: {
                'series': {
                    s_name: s_data.get('Values', [])
                    for s_name, s_data in chart.get('Series', {}).items()
                }
            }
            for name, chart in result.get('Charts', {}).items()
        },
        'orders': result.get('Orders', {}),
        'alpha_runtime_statistics': result.get('AlphaRuntimeStatistics', {})
    }
```

### Results Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

def analyze_backtest_results(client: LeanAPIClient,
                             project_id: int,
                             backtest_id: str) -> pd.DataFrame:
    """
    Comprehensive backtest analysis
    """
    results = client.read_backtest(project_id, backtest_id)
    data = extract_backtest_data(results)

    # Key statistics
    stats = data['statistics']
    print("=== Performance Summary ===")
    key_metrics = [
        'Total Net Profit', 'Sharpe Ratio', 'Sortino Ratio',
        'Drawdown', 'Win Rate', 'Profit-Loss Ratio',
        'Alpha', 'Beta', 'Annual Standard Deviation'
    ]
    for metric in key_metrics:
        print(f"{metric}: {stats.get(metric, 'N/A')}")

    # Extract equity curve
    equity_chart = data['charts'].get('Strategy Equity', {})
    equity_series = equity_chart.get('series', {}).get('Equity', [])

    if equity_series:
        equity_df = pd.DataFrame(equity_series)
        equity_df['time'] = pd.to_datetime(equity_df['x'], unit='s')
        equity_df.set_index('time', inplace=True)

        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity_df['y'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()

        return equity_df

    return pd.DataFrame()


def compare_backtests(client: LeanAPIClient,
                      project_id: int,
                      backtest_ids: List[str]) -> pd.DataFrame:
    """
    Compare multiple backtests
    """
    comparison = []

    for bt_id in backtest_ids:
        results = client.read_backtest(project_id, bt_id)
        stats = results.get('result', {}).get('Statistics', {})

        comparison.append({
            'backtest_id': bt_id,
            'name': results.get('name', bt_id),
            'total_return': stats.get('Total Net Profit', '0%'),
            'sharpe': stats.get('Sharpe Ratio', 0),
            'max_drawdown': stats.get('Drawdown', '0%'),
            'win_rate': stats.get('Win Rate', '0%'),
            'total_trades': results.get('result', {}).get('TotalOrders', 0)
        })

    return pd.DataFrame(comparison)
```

## Local Lean Engine

### Running Lean Locally

```bash
# Clone Lean engine
git clone https://github.com/QuantConnect/Lean.git
cd Lean

# Using Docker (recommended)
docker-compose up

# Or with lean-cli
pip install lean
lean init
lean backtest "My Project"
```

### Lean CLI Commands

```bash
# Initialize workspace
lean init

# Create new project
lean project-create "MyStrategy" --language python

# Run backtest locally
lean backtest "MyStrategy"

# Run with custom data
lean backtest "MyStrategy" --data-provider-historical "QuantConnect"

# Deploy to cloud
lean cloud push --project "MyStrategy"
lean cloud backtest "MyStrategy" --name "Test Run"

# Live trading
lean live "MyStrategy" --brokerage "Interactive Brokers"
```

## Best Practices

1. **Warm-up indicators**: Always use `SetWarmUp()` to ensure indicators have history
2. **Check data availability**: Verify `data.ContainsKey(symbol)` before accessing
3. **Universe hygiene**: Handle `OnSecuritiesChanged` to clean up removed securities
4. **Resolution matters**: Higher resolution = more data = slower backtests
5. **Scheduled events**: Prefer scheduled events over checking time in `OnData`
6. **Risk management**: Always include position sizing and stop-loss logic

## Common Pitfalls

- **Look-ahead bias**: Using `History()` with future data dates
- **Survivorship bias**: Universe only includes current stocks, not delisted
- **Not handling splits/dividends**: Use adjusted prices (default)
- **Over-trading**: Transaction costs destroy strategies with high turnover
- **Overfitting**: In-sample optimization without out-of-sample validation
- **Ignoring slippage**: Backtest fills are idealized; live trading has slippage

---

**Skill Type**: Finance - QuantConnect/Lean
**Complexity**: Advanced
**Typical Usage**: Algo trading development, backtesting, cloud deployment
