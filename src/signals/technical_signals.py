#!/usr/bin/env python3
"""
Technical Signal Generator
Generates momentum, volatility, and sector rotation signals
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TechnicalSignalGenerator:
    """Generates technical trading signals from market data."""
    
    def __init__(self):
        """Initialize the technical signal generator."""
        self.sector_map = {
            'AAPL': 'Technology Hardware',
            'MSFT': 'Software',
            'GOOGL': 'Internet Services',
            'AMZN': 'E-commerce',
            'META': 'Social Media',
            'NVDA': 'Semiconductors',
            'TSLA': 'Electric Vehicles',
            'AVGO': 'Semiconductors',
            'ORCL': 'Software',
            'ADBE': 'Software',
            'CRM': 'Software',
            'AMD': 'Semiconductors',
            'INTC': 'Semiconductors',
            'CSCO': 'Networking',
            'QCOM': 'Semiconductors',
            'NFLX': 'Streaming',
            'INTU': 'Software',
            'NOW': 'Software',
            'UBER': 'Ride Sharing',
            'SHOP': 'E-commerce',
            'SQ': 'Fintech',
            'PYPL': 'Fintech',
            'COIN': 'Fintech',
            'PLTR': 'Software',
            'SNAP': 'Social Media',
            'MU': 'Semiconductors',
            'LRCX': 'Semiconductors',
            'AMAT': 'Semiconductors',
            'MRVL': 'Semiconductors',
            'ABNB': 'Travel Tech'
        }
    
    def calculate_momentum_signals(self, df: pd.DataFrame, ticker: str) -> Dict[str, float]:
        """Generate momentum signals for a stock.
        
        Args:
            df: DataFrame with OHLCV and technical indicators
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with momentum signal components
        """
        if df.empty or len(df) < 50:
            return self._empty_momentum_signals()
        
        latest = df.iloc[-1]
        recent_20 = df.tail(20)
        
        signals = {}
        
        # 1. Price vs SMA signals
        price_vs_sma20 = latest['Price_to_SMA20'] - 1  # Convert to percentage above/below
        price_vs_sma50 = latest['Price_to_SMA50'] - 1
        
        # Normalize to 0-1 scale (clamp extreme values)
        signals['sma20_signal'] = max(0, min(1, (price_vs_sma20 + 0.1) / 0.2))  # -10% to +10%
        signals['sma50_signal'] = max(0, min(1, (price_vs_sma50 + 0.15) / 0.3))  # -15% to +15%
        
        # 2. Trend strength (consistency of direction)
        price_changes = recent_20['Close'].pct_change().dropna()
        positive_days = (price_changes > 0).sum()
        trend_consistency = positive_days / len(price_changes)
        signals['trend_consistency'] = trend_consistency
        
        # 3. Price momentum (rate of change)
        price_momentum_5d = (latest['Close'] / recent_20.iloc[-6]['Close'] - 1) if len(recent_20) >= 6 else 0
        price_momentum_20d = (latest['Close'] / df.iloc[-21]['Close'] - 1) if len(df) >= 21 else 0
        
        # Normalize momentum (-20% to +20% -> 0 to 1)
        signals['momentum_5d'] = max(0, min(1, (price_momentum_5d + 0.2) / 0.4))
        signals['momentum_20d'] = max(0, min(1, (price_momentum_20d + 0.2) / 0.4))
        
        # 4. Volume confirmation
        avg_volume_ratio = recent_20['Volume_Ratio'].mean()
        volume_trend = recent_20['Volume_Ratio'].rolling(5).mean().iloc[-1] / recent_20['Volume_Ratio'].rolling(5).mean().iloc[-6] if len(recent_20) >= 6 else 1
        
        # Volume signals (higher is better for confirmation)
        signals['volume_confirmation'] = max(0, min(1, avg_volume_ratio / 2.0))  # 0 to 2x -> 0 to 1
        signals['volume_trend'] = max(0, min(1, (volume_trend - 0.5) / 1.0 + 0.5))  # 0.5x to 1.5x -> 0 to 1
        
        # 5. RSI-based signals
        rsi = latest['RSI_14']
        
        # RSI momentum (distance from neutral 50)
        if rsi > 50:
            rsi_momentum = min(1, (rsi - 50) / 30)  # 50 to 80 -> 0 to 1
        else:
            rsi_momentum = max(0, rsi / 50)  # 0 to 50 -> 0 to 1
        
        signals['rsi_momentum'] = rsi_momentum
        
        # RSI mean reversion opportunity (oversold/overbought)
        if rsi < 30:
            signals['rsi_opportunity'] = (30 - rsi) / 30  # More oversold = higher opportunity
        elif rsi > 70:
            signals['rsi_opportunity'] = (rsi - 70) / 30  # More overbought = lower opportunity (inverted)
            signals['rsi_opportunity'] = 1 - signals['rsi_opportunity']
        else:
            signals['rsi_opportunity'] = 0.5  # Neutral zone
        
        return signals
    
    def calculate_momentum_strength(self, momentum_signals: Dict[str, float]) -> float:
        """Calculate overall momentum strength score.
        
        Args:
            momentum_signals: Dictionary of momentum signal components
            
        Returns:
            Momentum strength score (0-1)
        """
        if not momentum_signals:
            return 0.0
        
        # Weighted combination of momentum signals
        weights = {
            'sma20_signal': 0.25,
            'sma50_signal': 0.20,
            'trend_consistency': 0.20,
            'momentum_5d': 0.15,
            'momentum_20d': 0.10,
            'volume_confirmation': 0.05,
            'volume_trend': 0.05,
            'rsi_momentum': 0.0  # Separate signal, not part of momentum strength
        }
        
        score = 0.0
        total_weight = 0.0
        
        for signal, weight in weights.items():
            if signal in momentum_signals:
                score += momentum_signals[signal] * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def calculate_volatility_signals(self, df: pd.DataFrame, ticker: str) -> Dict[str, float]:
        """Generate volatility-adjusted signals.
        
        Args:
            df: DataFrame with market data
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with volatility signals
        """
        if df.empty or len(df) < 20:
            return {'volatility_regime': 0.5, 'volatility_opportunity': 0.5, 'atr_signal': 0.5}
        
        latest = df.iloc[-1]
        recent_20 = df.tail(20)
        
        signals = {}
        
        # 1. Volatility regime classification
        current_volatility = latest['Volatility_20']
        avg_volatility = recent_20['Volatility_20'].mean()
        
        # Low volatility (0-1 scale where 0.5 is average)
        vol_ratio = current_volatility / (avg_volatility + 1e-8)  # Avoid division by zero
        signals['volatility_regime'] = max(0, min(1, 1 - (vol_ratio - 0.5) / 1.0))  # Lower vol = higher score
        
        # 2. ATR-based opportunity signal
        atr_pct = latest['ATR_Percent']
        atr_20_avg = recent_20['ATR_Percent'].mean()
        
        # ATR opportunity (moderate ATR is good, extreme ATR is risky)
        atr_ratio = atr_pct / (atr_20_avg + 1e-8)
        if atr_ratio < 0.7:  # Very low volatility
            signals['atr_signal'] = 0.3  # Lower opportunity
        elif atr_ratio > 2.0:  # Very high volatility
            signals['atr_signal'] = 0.2  # High risk
        else:  # Moderate volatility
            signals['atr_signal'] = 0.8  # Good trading opportunity
        
        # 3. Volatility trend (is volatility increasing or decreasing?)
        vol_trend = recent_20['ATR_Percent'].rolling(5).mean().iloc[-1] / recent_20['ATR_Percent'].rolling(5).mean().iloc[-6] if len(recent_20) >= 6 else 1
        
        # Decreasing volatility is generally positive
        signals['volatility_opportunity'] = max(0, min(1, (2 - vol_trend) / 1.5))  # 0.5x to 2x -> 1 to 0
        
        return signals
    
    def calculate_sector_rotation_signals(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, float]]:
        """Calculate sector rotation signals.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            
        Returns:
            Dictionary mapping ticker to sector rotation signals
        """
        sector_performance = {}
        stock_sector_signals = {}
        
        # Calculate sector performance
        for ticker, df in stock_data.items():
            if df.empty or len(df) < 20:
                continue
                
            sector = self.sector_map.get(ticker, 'Other')
            
            # Calculate recent performance
            recent_return = (df.iloc[-1]['Close'] / df.iloc[-6]['Close'] - 1) if len(df) >= 6 else 0
            
            if sector not in sector_performance:
                sector_performance[sector] = []
            sector_performance[sector].append((ticker, recent_return))
        
        # Calculate average sector performance
        sector_averages = {}
        for sector, stocks in sector_performance.items():
            if stocks:
                avg_return = sum(ret for _, ret in stocks) / len(stocks)
                sector_averages[sector] = avg_return
        
        # Calculate relative performance signals
        for ticker, df in stock_data.items():
            if df.empty:
                stock_sector_signals[ticker] = {'sector_relative': 0.5, 'sector_momentum': 0.5}
                continue
            
            sector = self.sector_map.get(ticker, 'Other')
            sector_avg = sector_averages.get(sector, 0)
            
            # Stock vs sector performance
            stock_return = (df.iloc[-1]['Close'] / df.iloc[-6]['Close'] - 1) if len(df) >= 6 else 0
            relative_performance = stock_return - sector_avg
            
            # Normalize relative performance (-10% to +10% -> 0 to 1)
            sector_relative = max(0, min(1, (relative_performance + 0.1) / 0.2))
            
            # Sector momentum (how well is the sector doing overall?)
            all_sector_avg = sum(sector_averages.values()) / len(sector_averages) if sector_averages else 0
            sector_vs_market = sector_avg - all_sector_avg
            sector_momentum = max(0, min(1, (sector_vs_market + 0.05) / 0.1))
            
            stock_sector_signals[ticker] = {
                'sector_relative': sector_relative,
                'sector_momentum': sector_momentum
            }
        
        return stock_sector_signals
    
    def generate_all_signals(self, stock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate all technical signals for multiple stocks.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            
        Returns:
            DataFrame with all signal strengths for each stock
        """
        results = []
        
        # Calculate sector rotation signals for all stocks
        sector_signals = self.calculate_sector_rotation_signals(stock_data)
        
        for ticker, df in stock_data.items():
            logger.info(f"Generating signals for {ticker}")
            
            # Initialize result row
            result = {'ticker': ticker, 'timestamp': datetime.now()}
            
            if df.empty or len(df) < 20:
                # Fill with neutral signals
                result.update(self._empty_signals())
                results.append(result)
                continue
            
            # 1. Momentum signals
            momentum_signals = self.calculate_momentum_signals(df, ticker)
            momentum_strength = self.calculate_momentum_strength(momentum_signals)
            
            result.update(momentum_signals)
            result['momentum_strength'] = momentum_strength
            
            # 2. Volatility signals
            volatility_signals = self.calculate_volatility_signals(df, ticker)
            result.update(volatility_signals)
            
            # 3. Sector rotation signals
            sector_sigs = sector_signals.get(ticker, {'sector_relative': 0.5, 'sector_momentum': 0.5})
            result.update(sector_sigs)
            
            # 4. Current market data
            latest = df.iloc[-1]
            result.update({
                'current_price': latest['Close'],
                'rsi': latest['RSI_14'],
                'atr_percent': latest['ATR_Percent'],
                'volume_ratio': latest['Volume_Ratio'],
                'price_to_sma20': latest['Price_to_SMA20'],
                'price_to_sma50': latest['Price_to_SMA50']
            })
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def backtest_signals(self, 
                        stock_data: Dict[str, pd.DataFrame],
                        lookback_days: int = 252,
                        signal_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Backtest signal performance.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            lookback_days: Number of days to backtest
            signal_weights: Custom signal weights for testing
            
        Returns:
            Dictionary with backtest results
        """
        if signal_weights is None:
            signal_weights = {
                'momentum_strength': 0.4,
                'volatility_regime': 0.2,
                'sector_relative': 0.3,
                'rsi_opportunity': 0.1
            }
        
        all_returns = []
        signal_accuracies = []
        
        for ticker, df in stock_data.items():
            if len(df) < lookback_days + 50:  # Need extra data for indicators
                continue
            
            # Split data for backtesting
            test_start = len(df) - lookback_days
            historical_data = df.iloc[:test_start].copy()
            test_data = df.iloc[test_start:].copy()
            
            for i in range(len(test_data) - 5):  # Need 5 days forward return
                current_idx = test_start + i
                signal_data = df.iloc[:current_idx+1]  # Data up to current point
                
                if len(signal_data) < 50:
                    continue
                
                # Generate signals
                momentum_sigs = self.calculate_momentum_signals(signal_data, ticker)
                momentum_strength = self.calculate_momentum_strength(momentum_sigs)
                volatility_sigs = self.calculate_volatility_signals(signal_data, ticker)
                
                # Calculate combined signal
                combined_signal = (
                    momentum_strength * signal_weights.get('momentum_strength', 0.4) +
                    volatility_sigs['volatility_regime'] * signal_weights.get('volatility_regime', 0.2) +
                    momentum_sigs.get('rsi_opportunity', 0.5) * signal_weights.get('rsi_opportunity', 0.1)
                )
                
                # Calculate forward return (5 days)
                current_price = signal_data.iloc[-1]['Close']
                future_price = df.iloc[current_idx + 5]['Close'] if current_idx + 5 < len(df) else current_price
                forward_return = (future_price / current_price - 1)
                
                all_returns.append(forward_return)
                
                # Signal accuracy (did high signal predict positive return?)
                if combined_signal > 0.6 and forward_return > 0:
                    signal_accuracies.append(1)
                elif combined_signal < 0.4 and forward_return < 0:
                    signal_accuracies.append(1)
                else:
                    signal_accuracies.append(0)
        
        if not all_returns:
            return {'accuracy': 0, 'avg_return': 0, 'sharpe_ratio': 0, 'total_trades': 0}
        
        accuracy = sum(signal_accuracies) / len(signal_accuracies) if signal_accuracies else 0
        avg_return = sum(all_returns) / len(all_returns)
        return_std = np.std(all_returns) if len(all_returns) > 1 else 1
        sharpe_ratio = (avg_return / return_std) * np.sqrt(252) if return_std > 0 else 0
        
        return {
            'accuracy': accuracy,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(all_returns),
            'return_std': return_std
        }
    
    def _empty_momentum_signals(self) -> Dict[str, float]:
        """Return empty momentum signals."""
        return {
            'sma20_signal': 0.5,
            'sma50_signal': 0.5,
            'trend_consistency': 0.5,
            'momentum_5d': 0.5,
            'momentum_20d': 0.5,
            'volume_confirmation': 0.5,
            'volume_trend': 0.5,
            'rsi_momentum': 0.5,
            'rsi_opportunity': 0.5
        }
    
    def _empty_signals(self) -> Dict[str, float]:
        """Return empty signals for stocks with insufficient data."""
        empty = self._empty_momentum_signals()
        empty.update({
            'momentum_strength': 0.5,
            'volatility_regime': 0.5,
            'volatility_opportunity': 0.5,
            'atr_signal': 0.5,
            'sector_relative': 0.5,
            'sector_momentum': 0.5,
            'current_price': 0,
            'rsi': 50,
            'atr_percent': 2.0,
            'volume_ratio': 1.0,
            'price_to_sma20': 1.0,
            'price_to_sma50': 1.0
        })
        return empty


def main():
    """Main function for standalone testing."""
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_collector import MarketDataCollector
    
    # Initialize components
    signal_gen = TechnicalSignalGenerator()
    data_collector = MarketDataCollector()
    
    # Load some test data
    tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META']
    stock_data = {}
    
    for ticker in tickers:
        df = data_collector.load_from_parquet(ticker)
        if df is not None and not df.empty:
            stock_data[ticker] = df
    
    if not stock_data:
        print("No market data found. Run data collection first.")
        return
    
    # Generate signals
    print("Generating technical signals...")
    signals_df = signal_gen.generate_all_signals(stock_data)
    
    # Display results
    print("\nTechnical Signal Results:")
    print("-" * 80)
    
    for _, row in signals_df.iterrows():
        print(f"\n{row['ticker']}:")
        print(f"  Momentum Strength: {row['momentum_strength']:.3f}")
        print(f"  Volatility Regime: {row['volatility_regime']:.3f}")
        print(f"  Sector Relative: {row['sector_relative']:.3f}")
        print(f"  RSI: {row['rsi']:.1f} | Price/SMA20: {row['price_to_sma20']:.3f}")
    
    # Backtest signals
    print("\nBacktesting signals...")
    backtest_results = signal_gen.backtest_signals(stock_data)
    
    print(f"\nBacktest Results:")
    print(f"  Accuracy: {backtest_results['accuracy']:.3f}")
    print(f"  Average Return: {backtest_results['avg_return']:.4f}")
    print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.3f}")
    print(f"  Total Trades: {backtest_results['total_trades']}")


if __name__ == "__main__":
    main()