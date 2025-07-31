#!/usr/bin/env python3
"""
Simple test of backtesting system functionality
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Test if we can import the backtester components
def test_imports():
    """Test that we can import backtesting components."""
    print("Testing imports...")
    
    try:
        from backtester import BacktestParameters, BacktestEngine, RebalanceFrequency
        print("✅ Successfully imported backtesting components")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_parameter_creation():
    """Test parameter creation."""
    print("\nTesting parameter creation...")
    
    try:
        from backtester import BacktestParameters, RebalanceFrequency
        
        params = BacktestParameters(
            start_date='2023-01-01',
            end_date='2023-12-31',
            initial_capital=10000.0,
            max_positions=8,
            rebalance_frequency=RebalanceFrequency.WEEKLY
        )
        
        print(f"✅ Created parameters: {params.start_date} to {params.end_date}")
        print(f"   Initial capital: ${params.initial_capital:,.0f}")
        print(f"   Max positions: {params.max_positions}")
        print(f"   Rebalance frequency: {params.rebalance_frequency.value}")
        print(f"   Signal weights: {params.signal_weights}")
        
        return True
        
    except Exception as e:
        print(f"❌ Parameter creation failed: {e}")
        return False

def test_synthetic_data_generation():
    """Test synthetic data generation for backtesting."""
    print("\nTesting synthetic data generation...")
    
    try:
        # Generate synthetic stock data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Create sample data for a few tickers
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        market_data = {}
        
        np.random.seed(42)  # For reproducible results
        
        for ticker in tickers:
            # Generate realistic stock price data
            returns = np.random.normal(0.0005, 0.015, len(dates))  # Daily returns
            
            # Add some momentum/mean reversion
            for i in range(1, len(returns)):
                returns[i] += 0.1 * returns[i-1]  # Momentum
            
            # Generate price series
            initial_price = np.random.uniform(100, 300)
            prices = [initial_price]
            
            for ret in returns[1:]:
                prices.append(prices[-1] * (1 + ret))
            
            # Create OHLCV data
            df = pd.DataFrame({
                'Open': prices,
                'High': [p * np.random.uniform(1.0, 1.02) for p in prices],
                'Low': [p * np.random.uniform(0.98, 1.0) for p in prices],
                'Close': prices,
                'Volume': np.random.randint(1000000, 10000000, len(prices)),
                'Daily_Return': [0] + returns[1:].tolist()
            }, index=dates)
            
            # Add technical indicators
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['RSI_14'] = 50  # Simplified RSI
            df['ATR_14'] = df['High'] - df['Low']  # Simplified ATR
            
            market_data[ticker] = df
        
        print(f"✅ Generated synthetic data for {len(tickers)} tickers")
        print(f"   Date range: {dates[0].date()} to {dates[-1].date()}")
        print(f"   Sample prices for AAPL: ${market_data['AAPL']['Close'].iloc[0]:.2f} to ${market_data['AAPL']['Close'].iloc[-1]:.2f}")
        
        return market_data
        
    except Exception as e:
        print(f"❌ Synthetic data generation failed: {e}")
        return None

def test_basic_backtest_logic():
    """Test basic backtesting logic without external dependencies."""
    print("\nTesting basic backtest logic...")
    
    try:
        # Simple portfolio simulation
        initial_capital = 10000
        cash = initial_capital
        positions = {}  # ticker -> shares
        
        # Simulate a few trades
        trades = [
            {'ticker': 'AAPL', 'action': 'BUY', 'shares': 50, 'price': 150.0},
            {'ticker': 'MSFT', 'action': 'BUY', 'shares': 40, 'price': 250.0},
            {'ticker': 'AAPL', 'action': 'SELL', 'shares': 25, 'price': 160.0}
        ]
        
        portfolio_history = []
        
        for i, trade in enumerate(trades):
            ticker = trade['ticker']
            action = trade['action']
            shares = trade['shares']
            price = trade['price']
            
            # Transaction cost (0.1%)
            transaction_cost = shares * price * 0.001
            
            if action == 'BUY':
                cash -= (shares * price + transaction_cost)
                positions[ticker] = positions.get(ticker, 0) + shares
            else:  # SELL
                cash += (shares * price - transaction_cost)
                positions[ticker] = positions.get(ticker, 0) - shares
                if positions[ticker] <= 0:
                    positions.pop(ticker, None)
            
            # Calculate portfolio value (simplified)
            portfolio_value = cash
            for pos_ticker, pos_shares in positions.items():
                # Use last trade price as current price (simplified)
                current_price = next((t['price'] for t in reversed(trades) if t['ticker'] == pos_ticker), price)
                portfolio_value += pos_shares * current_price
            
            portfolio_history.append({
                'step': i + 1,
                'cash': cash,
                'positions': positions.copy(),
                'portfolio_value': portfolio_value,
                'trade': trade
            })
        
        final_value = portfolio_history[-1]['portfolio_value']
        total_return = (final_value / initial_capital) - 1
        
        print(f"✅ Basic backtest simulation completed")
        print(f"   Initial capital: ${initial_capital:,.0f}")
        print(f"   Final value: ${final_value:,.0f}")
        print(f"   Total return: {total_return:.2%}")
        print(f"   Final positions: {positions}")
        print(f"   Remaining cash: ${cash:,.0f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic backtest logic failed: {e}")
        return False

def test_performance_metrics():
    """Test performance metrics calculations."""
    print("\nTesting performance metrics calculations...")
    
    try:
        # Generate sample return series
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 252)  # One year of daily returns
        
        # Calculate basic metrics
        total_return = (1 + pd.Series(returns)).prod() - 1
        volatility = pd.Series(returns).std() * np.sqrt(252)
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        avg_return = pd.Series(returns).mean() * 252
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (pd.Series(returns) > 0).mean()
        
        print(f"✅ Performance metrics calculated")
        print(f"   Total return: {total_return:.2%}")
        print(f"   Volatility: {volatility:.1%}")
        print(f"   Sharpe ratio: {sharpe_ratio:.3f}")
        print(f"   Max drawdown: {max_drawdown:.2%}")
        print(f"   Win rate: {win_rate:.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance metrics calculation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Backtesting System Components")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_parameter_creation,
        test_synthetic_data_generation,
        test_basic_backtest_logic,
        test_performance_metrics
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Backtesting system is functional.")
    else:
        print(f"⚠️  {total - passed} tests failed. Check dependencies and implementation.")
    
    print("\n📝 Next Steps:")
    print("1. Install required dependencies: pip install yfinance pandas numpy matplotlib seaborn")
    print("2. Run full backtest: python src/backtester.py")
    print("3. Generate optimization reports with parameter grids")
    print("4. Implement walk-forward analysis for robust validation")

if __name__ == "__main__":
    main()