#!/usr/bin/env python3
"""
Comprehensive Backtesting System
Tests trading strategies with realistic constraints and costs
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json
import itertools
import warnings
from dataclasses import dataclass, asdict
from enum import Enum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import sqlite3

# Plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available - plotting disabled")

# Local imports
from data_collector import MarketDataCollector
from signals.signal_aggregator import SignalAggregator
from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from llm_analyst import LLMAnalyst
from news_collector import NewsCollector
from performance_tracker import PerformanceTracker, PerformanceMetrics

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class RebalanceFrequency(Enum):
    """Rebalancing frequencies for testing."""
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


@dataclass
class BacktestParameters:
    """Container for backtest parameters."""
    start_date: str
    end_date: str
    initial_capital: float = 10000.0
    max_positions: int = 8
    rebalance_frequency: RebalanceFrequency = RebalanceFrequency.WEEKLY
    transaction_cost_pct: float = 0.001  # 0.1% per trade
    slippage_pct: float = 0.0005  # 0.05% slippage
    signal_weights: Dict[str, float] = None
    position_limit_pct: float = 0.15  # Max 15% per position
    cash_buffer_pct: float = 0.10  # Keep 10% cash buffer
    drift_threshold: float = 0.20  # 20% drift before rebalancing
    use_regime_adjustment: bool = True
    min_signal_score: float = 0.5
    
    def __post_init__(self):
        if self.signal_weights is None:
            self.signal_weights = {
                'llm_sentiment': 0.30,
                'technical': 0.40,
                'volatility': 0.20,
                'sector': 0.10
            }


@dataclass
class BacktestResult:
    """Container for backtest results."""
    parameters: BacktestParameters
    portfolio_history: pd.DataFrame
    trades: List[Dict[str, Any]]
    performance_metrics: PerformanceMetrics
    signal_performance: Dict[str, Any]
    risk_metrics: Dict[str, float]
    benchmark_comparison: Dict[str, float]
    execution_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result_dict = {
            'parameters': asdict(self.parameters),
            'portfolio_history': self.portfolio_history.to_dict('records'),
            'trades': self.trades,
            'performance_metrics': asdict(self.performance_metrics),
            'signal_performance': self.signal_performance,
            'risk_metrics': self.risk_metrics,
            'benchmark_comparison': self.benchmark_comparison,
            'execution_time': self.execution_time
        }
        
        # Convert enums to strings
        result_dict['parameters']['rebalance_frequency'] = self.parameters.rebalance_frequency.value
        
        return result_dict


class BacktestEngine:
    """Core backtesting engine with realistic trading simulation."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize backtest engine.
        
        Args:
            data_dir: Directory containing market data
        """
        self.data_dir = Path(data_dir)
        self.data_collector = MarketDataCollector()
        self.signal_aggregator = SignalAggregator()
        
        # Cache for market data
        self.market_data_cache: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        # Trading constraints
        self.min_trade_size = 1  # Minimum 1 share
        self.max_single_trade_pct = 0.20  # Max 20% of portfolio in single trade
        
        logger.info("Backtest engine initialized")
    
    def load_historical_data(self, 
                           tickers: List[str], 
                           start_date: str, 
                           end_date: str,
                           force_download: bool = False) -> Dict[str, pd.DataFrame]:
        """Load historical data for backtesting.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force_download: Force re-download of data
            
        Returns:
            Dictionary of ticker -> DataFrame
        """
        logger.info(f"Loading historical data for {len(tickers)} tickers: {start_date} to {end_date}")
        
        # Clear cache if force download
        if force_download:
            self.market_data_cache.clear()
        
        # Check cache first
        cache_key = f"{start_date}_{end_date}"
        if not force_download and cache_key in self.market_data_cache:
            return self.market_data_cache[cache_key]
        
        market_data = {}
        
        for ticker in tickers:
            try:
                # Load from parquet if available
                df = self.data_collector.load_from_parquet(ticker)
                
                if df is None or df.empty:
                    logger.warning(f"No data available for {ticker}")
                    continue
                
                # Filter by date range
                df['Date'] = pd.to_datetime(df.index)
                df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
                
                if df.empty:
                    logger.warning(f"No data in date range for {ticker}")
                    continue
                
                # Ensure required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {ticker}")
                    continue
                
                # Add technical indicators if not present
                if 'SMA_20' not in df.columns:
                    df = self.data_collector.add_technical_indicators(df)
                
                market_data[ticker] = df.copy()
                logger.debug(f"Loaded {len(df)} days of data for {ticker}")
                
            except Exception as e:
                logger.error(f"Error loading data for {ticker}: {e}")
                continue
        
        # Cache the results
        self.market_data_cache[cache_key] = market_data
        
        logger.info(f"Successfully loaded data for {len(market_data)} tickers")
        return market_data
    
    def generate_benchmark_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Generate benchmark (S&P 500) data for comparison.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with benchmark returns
        """
        if self.benchmark_data is not None:
            # Filter existing benchmark data
            benchmark_filtered = self.benchmark_data[
                (self.benchmark_data.index >= start_date) & 
                (self.benchmark_data.index <= end_date)
            ].copy()
            
            if not benchmark_filtered.empty:
                return benchmark_filtered
        
        # Generate synthetic S&P 500 data (in production, use real SPY data)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Realistic S&P 500 parameters
        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(0.0004, 0.01, len(dates))  # ~10% annual, 16% vol
        
        # Add some serial correlation
        for i in range(1, len(daily_returns)):
            daily_returns[i] += 0.05 * daily_returns[i-1]
        
        # Create price series
        initial_price = 4000  # Starting S&P level
        prices = [initial_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        benchmark_df = pd.DataFrame({
            'Price': prices,
            'Daily_Return': daily_returns,
            'Cumulative_Return': pd.Series(daily_returns).cumsum()
        }, index=dates)
        
        self.benchmark_data = benchmark_df
        return benchmark_df
    
    def simulate_trade_execution(self, 
                                ticker: str, 
                                shares: float, 
                                price: float, 
                                action: str,
                                parameters: BacktestParameters) -> Tuple[float, float]:
        """Simulate realistic trade execution with costs and slippage.
        
        Args:
            ticker: Stock ticker
            shares: Number of shares
            price: Base price
            action: 'BUY' or 'SELL'
            parameters: Backtest parameters
            
        Returns:
            Tuple of (executed_price, total_cost)
        """
        # Apply slippage (market impact)
        slippage_factor = 1 + parameters.slippage_pct if action == 'BUY' else 1 - parameters.slippage_pct
        executed_price = price * slippage_factor
        
        # Calculate trade value
        trade_value = abs(shares * executed_price)
        
        # Apply transaction costs
        transaction_cost = trade_value * parameters.transaction_cost_pct
        
        # Minimum transaction cost (e.g., $1 per trade)
        transaction_cost = max(transaction_cost, 1.0)
        
        return executed_price, transaction_cost
    
    def run_backtest(self, parameters: BacktestParameters) -> BacktestResult:
        """Run a complete backtest with given parameters.
        
        Args:
            parameters: Backtest parameters
            
        Returns:
            BacktestResult object
        """
        start_time = datetime.now()
        logger.info(f"Starting backtest: {parameters.start_date} to {parameters.end_date}")
        
        # Load market data
        tickers = MarketDataCollector.TECH_UNIVERSE
        market_data = self.load_historical_data(tickers, parameters.start_date, parameters.end_date)
        
        if not market_data:
            raise ValueError("No market data available for backtest period")
        
        # Generate benchmark data
        benchmark_data = self.generate_benchmark_data(parameters.start_date, parameters.end_date)
        
        # Initialize portfolio tracking
        portfolio_history = []
        trades = []
        current_positions = {}  # ticker -> shares
        cash = parameters.initial_capital
        total_transaction_costs = 0.0
        
        # Get all trading dates
        all_dates = set()
        for df in market_data.values():
            all_dates.update(df.index.date)
        trading_dates = sorted(list(all_dates))
        
        # Determine rebalancing dates
        rebalance_dates = self._get_rebalance_dates(
            trading_dates, parameters.rebalance_frequency
        )
        
        logger.info(f"Found {len(trading_dates)} trading days, {len(rebalance_dates)} rebalance dates")
        
        # Main backtesting loop
        for i, current_date in enumerate(trading_dates):
            current_date_str = current_date.isoformat()
            
            # Get current prices
            current_prices = {}
            for ticker, df in market_data.items():
                date_data = df[df.index.date == current_date]
                if not date_data.empty:
                    current_prices[ticker] = date_data['Close'].iloc[0]
            
            # Calculate current portfolio value
            portfolio_value = cash
            position_values = {}
            
            for ticker, shares in current_positions.items():
                if ticker in current_prices:
                    market_value = shares * current_prices[ticker]
                    portfolio_value += market_value
                    position_values[ticker] = market_value
            
            # Record portfolio snapshot
            portfolio_snapshot = {
                'date': current_date_str,
                'total_value': portfolio_value,
                'cash': cash,
                'positions': position_values.copy(),
                'num_positions': len([v for v in position_values.values() if v > 0]),
                'daily_return': 0.0  # Will be calculated later
            }
            
            # Calculate daily return
            if portfolio_history:
                prev_value = portfolio_history[-1]['total_value']
                portfolio_snapshot['daily_return'] = (portfolio_value / prev_value) - 1 if prev_value > 0 else 0
            
            portfolio_history.append(portfolio_snapshot)
            
            # Check if it's a rebalancing date
            if current_date in rebalance_dates:
                logger.debug(f"Rebalancing on {current_date_str}")
                
                # Generate trading signals for current date
                signals = self._generate_historical_signals(
                    market_data, current_date, parameters
                )
                
                if signals is not None and not signals.empty:
                    # Generate trades
                    new_trades, new_cash, transaction_costs = self._execute_rebalancing(
                        signals, current_positions, current_prices, cash, 
                        portfolio_value, current_date_str, parameters
                    )
                    
                    # Update portfolio state
                    for trade in new_trades:
                        ticker = trade['ticker']
                        shares_change = trade['shares'] if trade['action'] == 'BUY' else -trade['shares']
                        current_positions[ticker] = current_positions.get(ticker, 0) + shares_change
                        
                        # Remove positions that are now zero
                        if abs(current_positions[ticker]) < 0.001:
                            current_positions.pop(ticker, None)
                    
                    cash = new_cash
                    total_transaction_costs += transaction_costs
                    trades.extend(new_trades)
        
        # Convert portfolio history to DataFrame
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_backtest_metrics(
            portfolio_df, benchmark_data, parameters
        )
        
        # Calculate signal performance
        signal_performance = self._analyze_signal_performance(trades, market_data)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(portfolio_df, current_positions, market_data)
        
        # Benchmark comparison
        benchmark_comparison = self._compare_to_benchmark(portfolio_df, benchmark_data)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        result = BacktestResult(
            parameters=parameters,
            portfolio_history=portfolio_df,
            trades=trades,
            performance_metrics=performance_metrics,
            signal_performance=signal_performance,
            risk_metrics=risk_metrics,
            benchmark_comparison=benchmark_comparison,
            execution_time=execution_time
        )
        
        logger.info(f"Backtest completed in {execution_time:.1f}s. "
                   f"Total return: {performance_metrics.total_return:.2%}, "
                   f"Sharpe: {performance_metrics.sharpe_ratio:.3f}")
        
        return result
    
    def _get_rebalance_dates(self, 
                           trading_dates: List[datetime.date], 
                           frequency: RebalanceFrequency) -> List[datetime.date]:
        """Get rebalancing dates based on frequency.
        
        Args:
            trading_dates: All available trading dates
            frequency: Rebalancing frequency
            
        Returns:
            List of rebalancing dates
        """
        rebalance_dates = []
        
        if frequency == RebalanceFrequency.WEEKLY:
            # Every Friday (or last trading day of week)
            current_week = None
            for date in trading_dates:
                week = date.isocalendar()[1]  # Week number
                if current_week != week:
                    if current_week is not None:
                        rebalance_dates.append(date)
                    current_week = week
        
        elif frequency == RebalanceFrequency.BIWEEKLY:
            # Every other Friday
            week_count = 0
            current_week = None
            for date in trading_dates:
                week = date.isocalendar()[1]
                if current_week != week:
                    week_count += 1
                    if week_count % 2 == 0:
                        rebalance_dates.append(date)
                    current_week = week
        
        elif frequency == RebalanceFrequency.MONTHLY:
            # Last trading day of each month
            current_month = None
            for date in trading_dates:
                month = date.month
                if current_month != month:
                    if current_month is not None:
                        rebalance_dates.append(date)
                    current_month = month
        
        return rebalance_dates
    
    def _generate_historical_signals(self, 
                                   market_data: Dict[str, pd.DataFrame],
                                   current_date: datetime.date,
                                   parameters: BacktestParameters) -> Optional[pd.DataFrame]:
        """Generate trading signals for historical date.
        
        Args:
            market_data: Historical market data
            current_date: Current simulation date
            parameters: Backtest parameters
            
        Returns:
            DataFrame with signals or None
        """
        try:
            # Get data up to current date for each ticker
            signals_data = []
            
            for ticker, df in market_data.items():
                # Get data up to current date (look-ahead bias prevention)
                historical_data = df[df.index.date <= current_date]
                
                if len(historical_data) < 50:  # Need sufficient history
                    continue
                
                latest_data = historical_data.iloc[-1]
                
                # Generate technical signals
                technical_signal = self._calculate_technical_signal(historical_data)
                
                # Generate volatility signal
                volatility_signal = self._calculate_volatility_signal(historical_data)
                
                # For backtesting, use simplified LLM signal (would need historical news)
                llm_signal = 0.5  # Neutral for backtesting
                
                # Sector signal (simplified)
                sector_signal = self._calculate_sector_signal(ticker, market_data, current_date)
                
                # Calculate aggregate score
                weights = parameters.signal_weights
                aggregate_score = (
                    weights['technical'] * technical_signal +
                    weights['volatility'] * volatility_signal +
                    weights['llm_sentiment'] * llm_signal +
                    weights['sector'] * sector_signal
                )
                
                signals_data.append({
                    'ticker': ticker,
                    'technical_signal': technical_signal,
                    'volatility_signal': volatility_signal,
                    'llm_sentiment_signal': llm_signal,
                    'sector_signal': sector_signal,
                    'aggregate_score': aggregate_score,
                    'current_price': latest_data['Close'],
                    'volume': latest_data['Volume'],
                    'reasoning': f"Technical: {technical_signal:.3f}, Vol: {volatility_signal:.3f}"
                })
            
            if not signals_data:
                return None
            
            signals_df = pd.DataFrame(signals_data)
            
            # Filter by minimum score
            signals_df = signals_df[signals_df['aggregate_score'] >= parameters.min_signal_score]
            
            return signals_df.sort_values('aggregate_score', ascending=False)
        
        except Exception as e:
            logger.error(f"Error generating signals for {current_date}: {e}")
            return None
    
    def _calculate_technical_signal(self, df: pd.DataFrame) -> float:
        """Calculate technical signal strength."""
        try:
            latest = df.iloc[-1]
            
            # Price vs moving averages
            price = latest['Close']
            sma_20 = latest.get('SMA_20', price)
            sma_50 = latest.get('SMA_50', price)
            
            # RSI signal
            rsi = latest.get('RSI_14', 50)
            
            # Calculate signal
            ma_signal = 0.5
            if sma_20 > 0 and sma_50 > 0:
                ma_signal = 0.5 + 0.3 * ((price / sma_20) - 1) + 0.2 * ((sma_20 / sma_50) - 1)
            
            rsi_signal = 0.5
            if 30 <= rsi <= 70:
                rsi_signal = 0.5 + 0.3 * ((50 - rsi) / 20)  # Contrarian RSI
            
            signal = 0.7 * ma_signal + 0.3 * rsi_signal
            return np.clip(signal, 0, 1)
            
        except Exception:
            return 0.5
    
    def _calculate_volatility_signal(self, df: pd.DataFrame) -> float:
        """Calculate volatility-adjusted signal."""
        try:
            if len(df) < 20:
                return 0.5
            
            returns = df['Close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Lower volatility gets higher signal (risk-adjusted)
            if volatility < 0.15:  # Low vol
                return 0.7
            elif volatility < 0.25:  # Medium vol
                return 0.5
            else:  # High vol
                return 0.3
                
        except Exception:
            return 0.5
    
    def _calculate_sector_signal(self, 
                               ticker: str, 
                               market_data: Dict[str, pd.DataFrame],
                               current_date: datetime.date) -> float:
        """Calculate sector rotation signal."""
        try:
            # Simplified sector signal - compare to sector average
            sector_tickers = {
                'Tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
                'Semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM'],
                'Software': ['ORCL', 'ADBE', 'CRM', 'INTU', 'NOW']
            }
            
            # Find ticker's sector
            ticker_sector = None
            for sector, tickers in sector_tickers.items():
                if ticker in tickers:
                    ticker_sector = sector
                    break
            
            if not ticker_sector:
                return 0.5
            
            # Calculate sector performance vs overall
            sector_performance = []
            for sector_ticker in sector_tickers[ticker_sector]:
                if sector_ticker in market_data:
                    df = market_data[sector_ticker]
                    historical = df[df.index.date <= current_date]
                    if len(historical) >= 20:
                        recent_return = (historical['Close'].iloc[-1] / historical['Close'].iloc[-20]) - 1
                        sector_performance.append(recent_return)
            
            if sector_performance:
                avg_sector_return = np.mean(sector_performance)
                # Convert to signal (0-1)
                signal = 0.5 + np.clip(avg_sector_return * 5, -0.3, 0.3)  # Scale factor
                return signal
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def _execute_rebalancing(self, 
                           signals: pd.DataFrame,
                           current_positions: Dict[str, float],
                           current_prices: Dict[str, float],
                           cash: float,
                           portfolio_value: float,
                           date: str,
                           parameters: BacktestParameters) -> Tuple[List[Dict], float, float]:
        """Execute portfolio rebalancing.
        
        Returns:
            Tuple of (trades, updated_cash, transaction_costs)
        """
        trades = []
        total_transaction_costs = 0.0
        
        # Calculate target portfolio
        top_signals = signals.head(parameters.max_positions)
        
        # Calculate target weights (equal weight for simplicity, can be enhanced)
        available_capital = portfolio_value * (1 - parameters.cash_buffer_pct)
        target_weight_per_position = available_capital / len(top_signals) / portfolio_value
        target_weight_per_position = min(target_weight_per_position, parameters.position_limit_pct)
        
        target_positions = {}
        for _, signal in top_signals.iterrows():
            ticker = signal['ticker']
            if ticker in current_prices:
                target_value = portfolio_value * target_weight_per_position
                target_shares = target_value / current_prices[ticker]
                target_positions[ticker] = target_shares
        
        # Generate trades to reach target positions
        all_tickers = set(list(current_positions.keys()) + list(target_positions.keys()))
        
        for ticker in all_tickers:
            current_shares = current_positions.get(ticker, 0)
            target_shares = target_positions.get(ticker, 0)
            shares_diff = target_shares - current_shares
            
            if abs(shares_diff) < self.min_trade_size:
                continue
            
            if ticker not in current_prices:
                continue
            
            price = current_prices[ticker]
            action = 'BUY' if shares_diff > 0 else 'SELL'
            shares_to_trade = abs(shares_diff)
            
            # Check if we have enough cash for buys
            if action == 'BUY':
                estimated_cost = shares_to_trade * price * (1 + parameters.transaction_cost_pct)
                if estimated_cost > cash:
                    # Reduce trade size to fit available cash
                    shares_to_trade = cash / (price * (1 + parameters.transaction_cost_pct + parameters.slippage_pct))
                    if shares_to_trade < self.min_trade_size:
                        continue
            
            # Simulate trade execution
            executed_price, transaction_cost = self.simulate_trade_execution(
                ticker, shares_to_trade, price, action, parameters
            )
            
            # Update cash
            if action == 'BUY':
                cash_change = -(shares_to_trade * executed_price + transaction_cost)
            else:
                cash_change = shares_to_trade * executed_price - transaction_cost
            
            if action == 'BUY' and cash + cash_change < 0:
                continue  # Skip if insufficient funds
            
            cash += cash_change
            total_transaction_costs += transaction_cost
            
            # Record trade
            trade = {
                'date': date,
                'ticker': ticker,
                'action': action,
                'shares': shares_to_trade,
                'price': executed_price,
                'value': shares_to_trade * executed_price,
                'transaction_cost': transaction_cost,
                'signal_score': signals[signals['ticker'] == ticker]['aggregate_score'].iloc[0] if ticker in signals['ticker'].values else 0.5
            }
            trades.append(trade)
        
        return trades, cash, total_transaction_costs
    
    def _calculate_backtest_metrics(self, 
                                  portfolio_df: pd.DataFrame,
                                  benchmark_data: pd.DataFrame,
                                  parameters: BacktestParameters) -> PerformanceMetrics:
        """Calculate comprehensive backtest performance metrics."""
        
        if len(portfolio_df) < 2:
            # Return default metrics for insufficient data
            return PerformanceMetrics(
                date=parameters.end_date,
                period="backtest",
                total_return=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                trades_count=0,
                days_in_drawdown=0,
                recovery_time=None,
                benchmark_return=0.0,
                alpha=0.0,
                beta=0.0
            )
        
        returns = portfolio_df['daily_return'].dropna()
        
        # Basic metrics
        total_return = (portfolio_df['total_value'].iloc[-1] / portfolio_df['total_value'].iloc[0]) - 1
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Risk-adjusted metrics
        risk_free_rate = 0.02
        avg_return = returns.mean() * 252
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (avg_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        days_in_drawdown = (drawdown < -0.01).sum()
        
        # Recovery time
        recovery_time = None
        max_dd_idx = drawdown.idxmin()
        if max_dd_idx in drawdown.index:
            recovery_idx = drawdown[max_dd_idx:].ge(-0.001).idxmax()
            if recovery_idx and recovery_idx != max_dd_idx:
                recovery_time = (recovery_idx - max_dd_idx).days
        
        # Win rate
        winning_days = (returns > 0).sum()
        win_rate = winning_days / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        # Benchmark comparison
        benchmark_return = 0
        alpha = 0
        beta = 0
        
        if benchmark_data is not None and not benchmark_data.empty:
            # Align dates
            common_dates = portfolio_df.index.intersection(benchmark_data.index)
            if len(common_dates) > 10:
                port_returns = portfolio_df.loc[common_dates, 'daily_return']
                bench_returns = benchmark_data.loc[common_dates, 'Daily_Return']
                
                benchmark_return = bench_returns.sum()
                
                # Beta and alpha
                covariance = np.cov(port_returns, bench_returns)[0, 1]
                benchmark_variance = np.var(bench_returns)
                
                if benchmark_variance > 0:
                    beta = covariance / benchmark_variance
                    alpha = (port_returns.mean() - risk_free_rate/252) - beta * (bench_returns.mean() - risk_free_rate/252)
                    alpha *= 252  # Annualize
        
        return PerformanceMetrics(
            date=parameters.end_date,
            period="backtest",
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades_count=0,  # Will be updated by caller
            days_in_drawdown=days_in_drawdown,
            recovery_time=recovery_time,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta
        )
    
    def _analyze_signal_performance(self, 
                                  trades: List[Dict[str, Any]], 
                                  market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Analyze performance by signal type."""
        if not trades:
            return {'total_trades': 0}
        
        # Group trades by signal strength
        high_signal_trades = [t for t in trades if t.get('signal_score', 0) > 0.7]
        medium_signal_trades = [t for t in trades if 0.5 <= t.get('signal_score', 0) <= 0.7]
        low_signal_trades = [t for t in trades if t.get('signal_score', 0) < 0.5]
        
        return {
            'total_trades': len(trades),
            'high_signal_trades': len(high_signal_trades),
            'medium_signal_trades': len(medium_signal_trades),
            'low_signal_trades': len(low_signal_trades),
            'avg_signal_score': np.mean([t.get('signal_score', 0) for t in trades])
        }
    
    def _calculate_risk_metrics(self, 
                              portfolio_df: pd.DataFrame,
                              final_positions: Dict[str, float],
                              market_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Calculate risk metrics."""
        if portfolio_df.empty:
            return {}
        
        returns = portfolio_df['daily_return'].dropna()
        
        # VaR calculation (95% confidence)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        
        # Maximum portfolio concentration
        if 'positions' in portfolio_df.columns:
            max_concentration = 0
            for positions in portfolio_df['positions']:
                if positions and portfolio_df.loc[portfolio_df['positions'] == positions, 'total_value'].iloc[0] > 0:
                    total_value = portfolio_df.loc[portfolio_df['positions'] == positions, 'total_value'].iloc[0]
                    max_weight = max(positions.values()) / total_value if positions.values() else 0
                    max_concentration = max(max_concentration, max_weight)
        else:
            max_concentration = 0
        
        return {
            'var_95': var_95,
            'max_concentration': max_concentration,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 1 else 0,
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252) if len(returns[returns < 0]) > 1 else 0
        }
    
    def _compare_to_benchmark(self, 
                            portfolio_df: pd.DataFrame,
                            benchmark_data: pd.DataFrame) -> Dict[str, float]:
        """Compare portfolio performance to benchmark."""
        if portfolio_df.empty or benchmark_data is None or benchmark_data.empty:
            return {}
        
        # Calculate portfolio cumulative return
        portfolio_return = (portfolio_df['total_value'].iloc[-1] / portfolio_df['total_value'].iloc[0]) - 1
        
        # Calculate benchmark return for same period
        common_dates = portfolio_df.index.intersection(benchmark_data.index)
        if len(common_dates) > 1:
            benchmark_return = (benchmark_data.loc[common_dates, 'Price'].iloc[-1] / 
                              benchmark_data.loc[common_dates, 'Price'].iloc[0]) - 1
        else:
            benchmark_return = 0
        
        return {
            'portfolio_return': portfolio_return,
            'benchmark_return': benchmark_return,
            'excess_return': portfolio_return - benchmark_return,
            'tracking_error': 0  # Would need aligned daily returns
        }


class ParameterOptimizer:
    """Optimizes backtest parameters using grid search and walk-forward analysis."""
    
    def __init__(self, backtest_engine: BacktestEngine):
        """Initialize parameter optimizer.
        
        Args:
            backtest_engine: BacktestEngine instance
        """
        self.engine = backtest_engine
        self.results_cache = {}
    
    def grid_search(self, 
                   base_parameters: BacktestParameters,
                   parameter_grid: Dict[str, List[Any]],
                   optimization_metric: str = 'sharpe_ratio',
                   n_jobs: int = None) -> List[BacktestResult]:
        """Perform grid search optimization.
        
        Args:
            base_parameters: Base parameters to modify
            parameter_grid: Dictionary of parameter -> list of values to test
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            n_jobs: Number of parallel jobs (None for auto)
            
        Returns:
            List of BacktestResult objects sorted by optimization metric
        """
        logger.info(f"Starting grid search with {len(parameter_grid)} parameters")
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        combinations = list(itertools.product(*param_values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        # Prepare parameter sets
        parameter_sets = []
        for combo in combinations:
            params = BacktestParameters(**asdict(base_parameters))
            
            for i, param_name in enumerate(param_names):
                if param_name == 'signal_weights':
                    params.signal_weights = combo[i]
                else:
                    setattr(params, param_name, combo[i])
            
            parameter_sets.append(params)
        
        # Run backtests in parallel
        if n_jobs is None:
            n_jobs = min(mp.cpu_count() - 1, len(parameter_sets))
        
        results = []
        
        if n_jobs > 1:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                future_to_params = {
                    executor.submit(self._run_single_backtest, params): params 
                    for params in parameter_sets
                }
                
                for future in as_completed(future_to_params):
                    try:
                        result = future.result()
                        if result:
                            results.append(result)
                            logger.debug(f"Completed backtest {len(results)}/{len(parameter_sets)}")
                    except Exception as e:
                        logger.error(f"Backtest failed: {e}")
        else:
            # Sequential execution
            for i, params in enumerate(parameter_sets):
                try:
                    result = self._run_single_backtest(params)
                    if result:
                        results.append(result)
                    logger.debug(f"Completed backtest {i+1}/{len(parameter_sets)}")
                except Exception as e:
                    logger.error(f"Backtest {i+1} failed: {e}")
        
        # Sort by optimization metric
        results.sort(key=lambda x: getattr(x.performance_metrics, optimization_metric), reverse=True)
        
        logger.info(f"Grid search completed. Best {optimization_metric}: "
                   f"{getattr(results[0].performance_metrics, optimization_metric):.4f}")
        
        return results
    
    def walk_forward_analysis(self, 
                            parameters: BacktestParameters,
                            training_months: int = 12,
                            test_months: int = 3,
                            step_months: int = 1) -> Dict[str, Any]:
        """Perform walk-forward analysis.
        
        Args:
            parameters: Base parameters for testing
            training_months: Months of training data
            test_months: Months of test data
            step_months: Step size in months
            
        Returns:
            Dictionary with walk-forward results
        """
        logger.info(f"Starting walk-forward analysis: {training_months}m training, {test_months}m test")
        
        start_date = datetime.fromisoformat(parameters.start_date)
        end_date = datetime.fromisoformat(parameters.end_date)
        
        walk_forward_results = []
        current_date = start_date
        
        while current_date + timedelta(days=30 * (training_months + test_months)) <= end_date:
            # Define training period
            train_start = current_date
            train_end = current_date + timedelta(days=30 * training_months)
            
            # Define test period
            test_start = train_end + timedelta(days=1)
            test_end = train_end + timedelta(days=30 * test_months)
            
            logger.info(f"Walk-forward: Train {train_start.date()} to {train_end.date()}, "
                       f"Test {test_start.date()} to {test_end.date()}")
            
            try:
                # Training phase (parameter optimization)
                train_params = BacktestParameters(**asdict(parameters))
                train_params.start_date = train_start.isoformat()
                train_params.end_date = train_end.isoformat()
                
                # Simple parameter grid for walk-forward
                simple_grid = {
                    'signal_weights': [
                        {'llm_sentiment': 0.20, 'technical': 0.50, 'volatility': 0.20, 'sector': 0.10},
                        {'llm_sentiment': 0.30, 'technical': 0.40, 'volatility': 0.20, 'sector': 0.10},
                        {'llm_sentiment': 0.40, 'technical': 0.30, 'volatility': 0.20, 'sector': 0.10}
                    ]
                }
                
                # Quick optimization on training data
                train_results = self.grid_search(train_params, simple_grid, n_jobs=1)
                
                if not train_results:
                    logger.warning(f"No training results for period {train_start.date()}")
                    current_date += timedelta(days=30 * step_months)
                    continue
                
                best_params = train_results[0].parameters
                
                # Test phase
                test_params = BacktestParameters(**asdict(best_params))
                test_params.start_date = test_start.isoformat()
                test_params.end_date = test_end.isoformat()
                
                test_result = self._run_single_backtest(test_params)
                
                if test_result:
                    walk_forward_results.append({
                        'train_period': f"{train_start.date()} to {train_end.date()}",
                        'test_period': f"{test_start.date()} to {test_end.date()}",
                        'best_train_params': asdict(best_params),
                        'test_result': test_result.to_dict(),
                        'test_return': test_result.performance_metrics.total_return,
                        'test_sharpe': test_result.performance_metrics.sharpe_ratio
                    })
                
            except Exception as e:
                logger.error(f"Walk-forward analysis failed for period {current_date.date()}: {e}")
            
            # Move to next period
            current_date += timedelta(days=30 * step_months)
        
        # Aggregate results
        if walk_forward_results:
            avg_return = np.mean([r['test_return'] for r in walk_forward_results])
            avg_sharpe = np.mean([r['test_sharpe'] for r in walk_forward_results])
            
            summary = {
                'periods_tested': len(walk_forward_results),
                'average_return': avg_return,
                'average_sharpe': avg_sharpe,
                'return_std': np.std([r['test_return'] for r in walk_forward_results]),
                'sharpe_std': np.std([r['test_sharpe'] for r in walk_forward_results]),
                'periods': walk_forward_results
            }
        else:
            summary = {
                'periods_tested': 0,
                'average_return': 0,
                'average_sharpe': 0,
                'periods': []
            }
        
        logger.info(f"Walk-forward analysis completed. Average return: {summary['average_return']:.2%}, "
                   f"Average Sharpe: {summary['average_sharpe']:.3f}")
        
        return summary
    
    def _run_single_backtest(self, parameters: BacktestParameters) -> Optional[BacktestResult]:
        """Run a single backtest (for parallel execution)."""
        try:
            # Create new engine instance for parallel execution
            engine = BacktestEngine()
            return engine.run_backtest(parameters)
        except Exception as e:
            logger.error(f"Single backtest failed: {e}")
            return None


class BacktestReporter:
    """Generates comprehensive backtest reports with charts."""
    
    def __init__(self, output_dir: str = "reports/backtests"):
        """Initialize backtest reporter.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_single_backtest_report(self, 
                                      result: BacktestResult,
                                      save_path: Optional[str] = None) -> str:
        """Generate comprehensive report for a single backtest.
        
        Args:
            result: BacktestResult object
            save_path: Optional path to save report
            
        Returns:
            Path to generated report
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"backtest_report_{timestamp}.html"
        
        # Generate HTML report
        html_content = self._create_html_report(result)
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        # Generate charts if plotting is available
        if PLOTTING_AVAILABLE:
            chart_dir = save_path.parent / f"{save_path.stem}_charts"
            chart_dir.mkdir(exist_ok=True)
            self._generate_charts(result, chart_dir)
        
        logger.info(f"Backtest report generated: {save_path}")
        return str(save_path)
    
    def generate_optimization_report(self, 
                                   results: List[BacktestResult],
                                   optimization_metric: str,
                                   save_path: Optional[str] = None) -> str:
        """Generate optimization comparison report.
        
        Args:
            results: List of BacktestResult objects
            optimization_metric: Metric that was optimized
            save_path: Optional path to save report
            
        Returns:
            Path to generated report
        """
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = self.output_dir / f"optimization_report_{timestamp}.html"
        
        html_content = self._create_optimization_html_report(results, optimization_metric)
        
        with open(save_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Optimization report generated: {save_path}")
        return str(save_path)
    
    def _create_html_report(self, result: BacktestResult) -> str:
        """Create HTML report for single backtest."""
        params = result.parameters
        metrics = result.performance_metrics
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #1f77b4; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
                .positive {{ color: #28a745; }}
                .negative {{ color: #dc3545; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Autonomous Trading System - Backtest Report</h1>
                <p>Period: {params.start_date} to {params.end_date}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Parameters</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Initial Capital</td><td>${params.initial_capital:,.0f}</td></tr>
                    <tr><td>Max Positions</td><td>{params.max_positions}</td></tr>
                    <tr><td>Rebalance Frequency</td><td>{params.rebalance_frequency.value}</td></tr>
                    <tr><td>Transaction Cost</td><td>{params.transaction_cost_pct:.3%}</td></tr>
                    <tr><td>Slippage</td><td>{params.slippage_pct:.3%}</td></tr>
                    <tr><td>Position Limit</td><td>{params.position_limit_pct:.1%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Performance Summary</h2>
                <div class="metric">
                    <strong>Total Return</strong><br>
                    <span class="{'positive' if metrics.total_return > 0 else 'negative'}">{metrics.total_return:.2%}</span>
                </div>
                <div class="metric">
                    <strong>Sharpe Ratio</strong><br>
                    <span class="{'positive' if metrics.sharpe_ratio > 0 else 'negative'}">{metrics.sharpe_ratio:.3f}</span>
                </div>
                <div class="metric">
                    <strong>Max Drawdown</strong><br>
                    <span class="negative">{metrics.max_drawdown:.2%}</span>
                </div>
                <div class="metric">
                    <strong>Volatility</strong><br>
                    <span>{metrics.volatility:.1%}</span>
                </div>
                <div class="metric">
                    <strong>Win Rate</strong><br>
                    <span>{metrics.win_rate:.1%}</span>
                </div>
                <div class="metric">
                    <strong>Profit Factor</strong><br>
                    <span>{metrics.profit_factor:.2f}</span>
                </div>
            </div>
            
            <div class="section">
                <h2>Risk Metrics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Sortino Ratio</td><td>{metrics.sortino_ratio:.3f}</td></tr>
                    <tr><td>Days in Drawdown</td><td>{metrics.days_in_drawdown}</td></tr>
                    <tr><td>Recovery Time (days)</td><td>{metrics.recovery_time or 'N/A'}</td></tr>
                    <tr><td>Beta</td><td>{metrics.beta:.3f}</td></tr>
                    <tr><td>Alpha</td><td>{metrics.alpha:.3f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Signal Weights</h2>
                <table>
                    <tr><th>Signal Type</th><th>Weight</th></tr>
        """
        
        for signal_type, weight in params.signal_weights.items():
            html += f"<tr><td>{signal_type.replace('_', ' ').title()}</td><td>{weight:.1%}</td></tr>"
        
        html += f"""
                </table>
            </div>
            
            <div class="section">
                <h2>Benchmark Comparison</h2>
                <table>
                    <tr><th>Metric</th><th>Portfolio</th><th>Benchmark</th><th>Excess</th></tr>
                    <tr>
                        <td>Total Return</td>
                        <td>{result.benchmark_comparison.get('portfolio_return', 0):.2%}</td>
                        <td>{result.benchmark_comparison.get('benchmark_return', 0):.2%}</td>
                        <td class="{'positive' if result.benchmark_comparison.get('excess_return', 0) > 0 else 'negative'}">
                            {result.benchmark_comparison.get('excess_return', 0):.2%}
                        </td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Trading Activity</h2>
                <p><strong>Total Trades:</strong> {len(result.trades)}</p>
                <p><strong>Signal Performance:</strong></p>
                <ul>
                    <li>High Signal Trades: {result.signal_performance.get('high_signal_trades', 0)}</li>
                    <li>Medium Signal Trades: {result.signal_performance.get('medium_signal_trades', 0)}</li>
                    <li>Low Signal Trades: {result.signal_performance.get('low_signal_trades', 0)}</li>
                    <li>Average Signal Score: {result.signal_performance.get('avg_signal_score', 0):.3f}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Execution Details</h2>
                <p><strong>Execution Time:</strong> {result.execution_time:.1f} seconds</p>
                <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_optimization_html_report(self, 
                                       results: List[BacktestResult], 
                                       optimization_metric: str) -> str:
        """Create HTML report for optimization results."""
        if not results:
            return "<html><body><h1>No results to display</h1></body></html>"
        
        best_result = results[0]
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Parameter Optimization Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #28a745; color: white; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .best {{ background-color: #d4edda; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Parameter Optimization Report</h1>
                <p>Optimization Metric: {optimization_metric.replace('_', ' ').title()}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Best Parameters</h2>
                <p><strong>Best {optimization_metric}:</strong> {getattr(best_result.performance_metrics, optimization_metric):.4f}</p>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Signal Weights - LLM</td><td>{best_result.parameters.signal_weights['llm_sentiment']:.1%}</td></tr>
                    <tr><td>Signal Weights - Technical</td><td>{best_result.parameters.signal_weights['technical']:.1%}</td></tr>
                    <tr><td>Signal Weights - Volatility</td><td>{best_result.parameters.signal_weights['volatility']:.1%}</td></tr>
                    <tr><td>Signal Weights - Sector</td><td>{best_result.parameters.signal_weights['sector']:.1%}</td></tr>
                    <tr><td>Max Positions</td><td>{best_result.parameters.max_positions}</td></tr>
                    <tr><td>Position Limit</td><td>{best_result.parameters.position_limit_pct:.1%}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>All Results (Top 10)</h2>
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>{optimization_metric.replace('_', ' ').title()}</th>
                        <th>Total Return</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                        <th>LLM Weight</th>
                        <th>Technical Weight</th>
                    </tr>
        """
        
        for i, result in enumerate(results[:10]):
            row_class = "best" if i == 0 else ""
            html += f"""
                    <tr class="{row_class}">
                        <td>{i+1}</td>
                        <td>{getattr(result.performance_metrics, optimization_metric):.4f}</td>
                        <td>{result.performance_metrics.total_return:.2%}</td>
                        <td>{result.performance_metrics.sharpe_ratio:.3f}</td>
                        <td>{result.performance_metrics.max_drawdown:.2%}</td>
                        <td>{result.parameters.signal_weights['llm_sentiment']:.1%}</td>
                        <td>{result.parameters.signal_weights['technical']:.1%}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _generate_charts(self, result: BacktestResult, chart_dir: Path) -> None:
        """Generate performance charts."""
        if not PLOTTING_AVAILABLE:
            return
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Portfolio value over time
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(result.portfolio_history.index, result.portfolio_history['total_value'], 
                linewidth=2, label='Portfolio Value')
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_dir / 'portfolio_value.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Daily returns histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        returns = result.portfolio_history['daily_return'].dropna()
        ax.hist(returns, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.3%}')
        ax.set_title('Daily Returns Distribution')
        ax.set_xlabel('Daily Return')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_dir / 'returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Drawdown chart
        if len(result.portfolio_history) > 1:
            fig, ax = plt.subplots(figsize=(12, 6))
            returns = result.portfolio_history['daily_return'].dropna()
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            
            ax.fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
            ax.plot(drawdown.index, drawdown, color='red', linewidth=1)
            ax.set_title('Portfolio Drawdown')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(chart_dir / 'drawdown.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Test the backtesting system."""
    print("Testing Comprehensive Backtesting System...")
    
    # Initialize backtester
    engine = BacktestEngine()
    
    # Define test parameters for 2023-2024
    base_params = BacktestParameters(
        start_date='2023-01-01',
        end_date='2024-12-31',
        initial_capital=10000.0,
        max_positions=8,
        rebalance_frequency=RebalanceFrequency.WEEKLY,
        transaction_cost_pct=0.001,
        slippage_pct=0.0005
    )
    
    print(f"Running backtest: {base_params.start_date} to {base_params.end_date}")
    
    # Run single backtest
    try:
        result = engine.run_backtest(base_params)
        
        print("\n=== BACKTEST RESULTS ===")
        print(f"Total Return: {result.performance_metrics.total_return:.2%}")
        print(f"Sharpe Ratio: {result.performance_metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {result.performance_metrics.max_drawdown:.2%}")
        print(f"Volatility: {result.performance_metrics.volatility:.1%}")
        print(f"Win Rate: {result.performance_metrics.win_rate:.1%}")
        print(f"Total Trades: {len(result.trades)}")
        print(f"Execution Time: {result.execution_time:.1f}s")
        
        # Generate report
        reporter = BacktestReporter()
        report_path = reporter.generate_single_backtest_report(result)
        print(f"\nDetailed report generated: {report_path}")
        
        # Test parameter optimization (small grid for demo)
        print("\n=== PARAMETER OPTIMIZATION ===")
        optimizer = ParameterOptimizer(engine)
        
        # Small parameter grid for testing
        test_grid = {
            'signal_weights': [
                {'llm_sentiment': 0.20, 'technical': 0.50, 'volatility': 0.20, 'sector': 0.10},
                {'llm_sentiment': 0.30, 'technical': 0.40, 'volatility': 0.20, 'sector': 0.10},
                {'llm_sentiment': 0.40, 'technical': 0.30, 'volatility': 0.20, 'sector': 0.10}
            ],
            'max_positions': [6, 8, 10]
        }
        
        # Adjust date range for faster testing
        test_params = BacktestParameters(**asdict(base_params))
        test_params.start_date = '2024-01-01'
        test_params.end_date = '2024-06-30'
        
        optimization_results = optimizer.grid_search(
            test_params, test_grid, optimization_metric='sharpe_ratio', n_jobs=1
        )
        
        print(f"Tested {len(optimization_results)} parameter combinations")
        if optimization_results:
            best = optimization_results[0]
            print(f"Best Sharpe Ratio: {best.performance_metrics.sharpe_ratio:.3f}")
            print(f"Best Signal Weights: {best.parameters.signal_weights}")
            
            # Generate optimization report
            opt_report_path = reporter.generate_optimization_report(
                optimization_results, 'sharpe_ratio'
            )
            print(f"Optimization report: {opt_report_path}")
        
        print("\n=== BACKTESTING SYSTEM TEST COMPLETED ===")
        
    except Exception as e:
        print(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()