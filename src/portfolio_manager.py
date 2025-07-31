#!/usr/bin/env python3
"""
Portfolio Management System
Tracks positions, handles rebalancing, and calculates performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types for trades."""
    BUY = "BUY"
    SELL = "SELL"


class RebalanceReason(Enum):
    """Reasons for rebalancing."""
    WEEKLY = "Weekly Rebalance"
    DRIFT = "Position Drift"
    SIGNAL = "New Signal"
    MANUAL = "Manual"


@dataclass
class Position:
    """Represents a stock position."""
    ticker: str
    shares: float
    avg_cost: float
    current_price: float
    market_value: float
    weight: float
    unrealized_pnl: float
    last_updated: str


@dataclass
class Trade:
    """Represents a trade execution."""
    trade_id: str
    timestamp: str
    ticker: str
    order_type: OrderType
    shares: float
    price: float
    gross_amount: float
    commission: float
    net_amount: float
    reason: RebalanceReason
    signal_score: Optional[float] = None


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""
    timestamp: str
    total_value: float
    cash: float
    invested_value: float
    positions: Dict[str, Position]
    daily_return: float
    cumulative_return: float


class PortfolioManager:
    """Manages portfolio positions, rebalancing, and performance tracking."""
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 max_position_size: float = 0.15,
                 min_cash_buffer: float = 0.10,
                 commission_rate: float = 0.001,
                 data_dir: str = "data/portfolio"):
        """Initialize portfolio manager.
        
        Args:
            initial_capital: Starting capital
            max_position_size: Maximum position size (15%)
            min_cash_buffer: Minimum cash buffer (10%)
            commission_rate: Commission rate per trade (0.1%)
            data_dir: Directory to store portfolio data
        """
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size
        self.min_cash_buffer = min_cash_buffer
        self.commission_rate = commission_rate
        
        # Current portfolio state
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.total_value = initial_capital
        
        # Data storage
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.trade_counter = 0
        
        # Performance tracking
        self.snapshots: List[PortfolioSnapshot] = []
        self.last_rebalance_date = None
        
        # Load existing data if available
        self._load_portfolio_state()
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        return f"T{datetime.now().strftime('%Y%m%d')}_{self.trade_counter:04d}"
    
    def _calculate_commission(self, gross_amount: float) -> float:
        """Calculate commission for a trade."""
        return abs(gross_amount) * self.commission_rate
    
    def update_prices(self, price_data: Dict[str, float]) -> None:
        """Update current prices for all positions.
        
        Args:
            price_data: Dictionary mapping ticker to current price
        """
        for ticker, position in self.positions.items():
            if ticker in price_data:
                position.current_price = price_data[ticker]
                position.market_value = position.shares * position.current_price
                position.unrealized_pnl = position.market_value - (position.shares * position.avg_cost)
                position.last_updated = datetime.now().isoformat()
        
        # Update total portfolio value
        self._update_total_value()
        
        # Update position weights
        self._update_position_weights()
    
    def _update_total_value(self) -> None:
        """Update total portfolio value."""
        invested_value = sum(pos.market_value for pos in self.positions.values())
        self.total_value = self.cash + invested_value
    
    def _update_position_weights(self) -> None:
        """Update position weights based on current market values."""
        if self.total_value <= 0:
            return
            
        for position in self.positions.values():
            position.weight = position.market_value / self.total_value
    
    def calculate_position_size(self, 
                              ticker: str,
                              signal_score: float,
                              volatility: float,
                              target_positions: int) -> float:
        """Calculate optimal position size.
        
        Args:
            ticker: Stock ticker
            signal_score: Signal strength (0-1)
            volatility: Stock volatility (annualized)
            target_positions: Target number of positions
            
        Returns:
            Target weight for the position
        """
        # Base equal weight
        base_weight = (1.0 - self.min_cash_buffer) / target_positions
        
        # Signal strength adjustment (0.8x to 1.2x)
        signal_adjustment = 0.8 + 0.4 * signal_score
        
        # Volatility adjustment (reduce size for high volatility)
        # Normal volatility around 20%, high volatility > 40%
        vol_adjustment = max(0.5, min(1.5, 1.0 - (volatility - 0.20) / 0.20))
        
        # Calculate adjusted weight
        target_weight = base_weight * signal_adjustment * vol_adjustment
        
        # Apply maximum position size constraint
        target_weight = min(target_weight, self.max_position_size)
        
        # Ensure we don't violate cash buffer
        max_investable = 1.0 - self.min_cash_buffer
        if target_weight > max_investable:
            target_weight = max_investable
        
        return target_weight
    
    def calculate_target_portfolio(self, 
                                 signals_df: pd.DataFrame,
                                 price_data: Dict[str, float],
                                 max_positions: int = 8) -> Dict[str, float]:
        """Calculate target portfolio weights.
        
        Args:
            signals_df: DataFrame with stock signals and scores
            price_data: Current price data
            max_positions: Maximum number of positions
            
        Returns:
            Dictionary mapping ticker to target weight
        """
        target_weights = {}
        
        # Select top signals
        top_signals = signals_df.head(max_positions)
        
        for _, row in top_signals.iterrows():
            ticker = row['ticker']
            signal_score = row['aggregate_score']
            
            # Get volatility (assume we have this in technical data)
            volatility = 0.25  # Default 25% annualized volatility
            if 'volatility' in row:
                volatility = row['volatility']
            elif ticker in price_data:
                # Rough volatility estimate from ATR if available
                volatility = 0.20  # Conservative default
            
            target_weight = self.calculate_position_size(
                ticker, signal_score, volatility, len(top_signals)
            )
            
            target_weights[ticker] = target_weight
        
        # Normalize weights to ensure they don't exceed investable amount
        total_target = sum(target_weights.values())
        max_investable = 1.0 - self.min_cash_buffer
        
        if total_target > max_investable:
            scale_factor = max_investable / total_target
            target_weights = {k: v * scale_factor for k, v in target_weights.items()}
        
        return target_weights
    
    def check_rebalance_needed(self, 
                             target_weights: Dict[str, float],
                             drift_threshold: float = 0.20) -> Tuple[bool, List[str]]:
        """Check if rebalancing is needed.
        
        Args:
            target_weights: Target portfolio weights
            drift_threshold: Maximum allowed drift (20%)
            
        Returns:
            Tuple of (rebalance_needed, reasons)
        """
        reasons = []
        
        # Check weekly rebalance
        if self.last_rebalance_date is None:
            reasons.append(RebalanceReason.WEEKLY.value)
        else:
            days_since_rebalance = (datetime.now() - datetime.fromisoformat(self.last_rebalance_date)).days
            if days_since_rebalance >= 7:
                reasons.append(RebalanceReason.WEEKLY.value)
        
        # Check position drift
        current_weights = {ticker: pos.weight for ticker, pos in self.positions.items()}
        
        for ticker, target_weight in target_weights.items():
            current_weight = current_weights.get(ticker, 0.0)
            
            # Calculate drift
            if target_weight > 0:
                drift = abs(current_weight - target_weight) / target_weight
                if drift > drift_threshold:
                    reasons.append(f"{RebalanceReason.DRIFT.value}: {ticker} ({drift:.1%})")
        
        # Check for positions not in target (should be sold)
        for ticker in current_weights:
            if ticker not in target_weights:
                reasons.append(f"{RebalanceReason.SIGNAL.value}: Remove {ticker}")
        
        return len(reasons) > 0, reasons
    
    def generate_rebalancing_orders(self, 
                                  target_weights: Dict[str, float],
                                  price_data: Dict[str, float],
                                  reason: RebalanceReason = RebalanceReason.WEEKLY) -> List[Dict[str, Any]]:
        """Generate orders to rebalance portfolio.
        
        Args:
            target_weights: Target portfolio weights
            price_data: Current price data
            reason: Reason for rebalancing
            
        Returns:
            List of order dictionaries
        """
        orders = []
        
        # Current position values
        current_weights = {ticker: pos.weight for ticker, pos in self.positions.items()}
        
        # Calculate target values
        target_values = {ticker: weight * self.total_value 
                        for ticker, weight in target_weights.items()}
        
        # Generate sell orders first (to free up capital)
        for ticker, position in self.positions.items():
            target_value = target_values.get(ticker, 0.0)
            current_value = position.market_value
            
            if target_value < current_value:
                # Need to sell some shares
                value_to_sell = current_value - target_value
                shares_to_sell = value_to_sell / position.current_price
                
                if shares_to_sell >= 0.01:  # Minimum trade size
                    orders.append({
                        'ticker': ticker,
                        'order_type': OrderType.SELL,
                        'shares': round(shares_to_sell, 4),
                        'price': position.current_price,
                        'reason': reason,
                        'priority': 1  # Sells first
                    })
            
            elif ticker not in target_weights:
                # Sell entire position
                orders.append({
                    'ticker': ticker,
                    'order_type': OrderType.SELL,
                    'shares': position.shares,
                    'price': position.current_price,
                    'reason': reason,
                    'priority': 1
                })
        
        # Generate buy orders
        for ticker, target_weight in target_weights.items():
            if ticker not in price_data:
                logger.warning(f"No price data for {ticker}, skipping")
                continue
                
            target_value = target_weight * self.total_value
            current_value = self.positions.get(ticker, Position(ticker, 0, 0, 0, 0, 0, 0, "")).market_value
            
            if target_value > current_value:
                # Need to buy more shares
                value_to_buy = target_value - current_value
                price = price_data[ticker]
                shares_to_buy = value_to_buy / price
                
                if shares_to_buy >= 0.01 and value_to_buy >= 10:  # Minimum trade size and value
                    orders.append({
                        'ticker': ticker,
                        'order_type': OrderType.BUY,
                        'shares': round(shares_to_buy, 4),
                        'price': price,
                        'reason': reason,
                        'priority': 2  # Buys second
                    })
        
        # Sort orders by priority (sells first, then buys)
        orders.sort(key=lambda x: x['priority'])
        
        return orders
    
    def execute_order(self, 
                     ticker: str,
                     order_type: OrderType,
                     shares: float,
                     price: float,
                     reason: RebalanceReason = RebalanceReason.MANUAL,
                     signal_score: Optional[float] = None) -> Trade:
        """Execute a trade order.
        
        Args:
            ticker: Stock ticker
            order_type: BUY or SELL
            shares: Number of shares
            price: Execution price
            reason: Reason for trade
            signal_score: Signal score if applicable
            
        Returns:
            Trade object
        """
        gross_amount = shares * price
        commission = self._calculate_commission(gross_amount)
        
        if order_type == OrderType.BUY:
            net_amount = -(gross_amount + commission)  # Negative cash flow
            
            # Check if we have enough cash
            if abs(net_amount) > self.cash:
                raise ValueError(f"Insufficient cash: need ${abs(net_amount):.2f}, have ${self.cash:.2f}")
            
            # Update or create position
            if ticker in self.positions:
                position = self.positions[ticker]
                total_cost = (position.shares * position.avg_cost) + gross_amount
                total_shares = position.shares + shares
                position.avg_cost = total_cost / total_shares
                position.shares = total_shares
            else:
                self.positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price,
                    current_price=price,
                    market_value=gross_amount,
                    weight=0.0,
                    unrealized_pnl=0.0,
                    last_updated=datetime.now().isoformat()
                )
            
        else:  # SELL
            net_amount = gross_amount - commission  # Positive cash flow
            
            if ticker not in self.positions:
                raise ValueError(f"Cannot sell {ticker}: no position exists")
            
            position = self.positions[ticker]
            if shares > position.shares:
                raise ValueError(f"Cannot sell {shares} shares of {ticker}: only have {position.shares}")
            
            # Update position
            position.shares -= shares
            if position.shares < 0.01:  # Close position if shares are negligible
                del self.positions[ticker]
            else:
                # Update market value for remaining shares
                position.market_value = position.shares * position.current_price
        
        # Update cash
        self.cash += net_amount
        
        # Create trade record
        trade = Trade(
            trade_id=self._generate_trade_id(),
            timestamp=datetime.now().isoformat(),
            ticker=ticker,
            order_type=order_type,
            shares=shares,
            price=price,
            gross_amount=gross_amount,
            commission=commission,
            net_amount=net_amount,
            reason=reason,
            signal_score=signal_score
        )
        
        self.trades.append(trade)
        
        # Update portfolio totals
        self._update_total_value()
        self._update_position_weights()
        
        logger.info(f"Executed trade: {order_type.value} {shares} shares of {ticker} at ${price:.2f}")
        
        return trade
    
    def execute_rebalancing_orders(self, orders: List[Dict[str, Any]]) -> List[Trade]:
        """Execute a list of rebalancing orders.
        
        Args:
            orders: List of order dictionaries
            
        Returns:
            List of executed trades
        """
        executed_trades = []
        
        for order in orders:
            try:
                trade = self.execute_order(
                    ticker=order['ticker'],
                    order_type=order['order_type'],
                    shares=order['shares'],
                    price=order['price'],
                    reason=order['reason']
                )
                executed_trades.append(trade)
                
            except Exception as e:
                logger.error(f"Failed to execute order for {order['ticker']}: {e}")
        
        # Update last rebalance date
        self.last_rebalance_date = datetime.now().isoformat()
        
        # Save portfolio state
        self._save_portfolio_state()
        
        return executed_trades
    
    def take_snapshot(self) -> PortfolioSnapshot:
        """Take a snapshot of current portfolio state.
        
        Returns:
            PortfolioSnapshot object
        """
        invested_value = sum(pos.market_value for pos in self.positions.values())
        
        # Calculate daily return
        daily_return = 0.0
        cumulative_return = (self.total_value / self.initial_capital) - 1.0
        
        if len(self.snapshots) > 0:
            prev_value = self.snapshots[-1].total_value
            daily_return = (self.total_value / prev_value) - 1.0 if prev_value > 0 else 0.0
        
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now().isoformat(),
            total_value=self.total_value,
            cash=self.cash,
            invested_value=invested_value,
            positions={ticker: pos for ticker, pos in self.positions.items()},
            daily_return=daily_return,
            cumulative_return=cumulative_return
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only last 365 snapshots
        if len(self.snapshots) > 365:
            self.snapshots = self.snapshots[-365:]
        
        return snapshot
    
    def calculate_performance_metrics(self, days_back: int = 252) -> Dict[str, float]:
        """Calculate portfolio performance metrics.
        
        Args:
            days_back: Number of days to look back for calculations
            
        Returns:
            Dictionary with performance metrics
        """
        if len(self.snapshots) < 2:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_trade_return': 0.0
            }
        
        # Get recent snapshots
        recent_snapshots = self.snapshots[-min(days_back, len(self.snapshots)):]
        
        # Calculate returns
        values = [s.total_value for s in recent_snapshots]
        returns = [values[i] / values[i-1] - 1 for i in range(1, len(values))]
        
        if not returns:
            return {}
        
        # Performance metrics
        total_return = (values[-1] / values[0]) - 1 if values[0] > 0 else 0
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        peak = values[0]
        max_drawdown = 0
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Trade metrics
        completed_trades = [t for t in self.trades if t.order_type == OrderType.SELL]
        win_rate = 0
        avg_trade_return = 0
        
        if completed_trades:
            # Simplified P&L calculation
            profitable_trades = sum(1 for t in completed_trades if t.net_amount > t.gross_amount * 0.001)
            win_rate = profitable_trades / len(completed_trades)
            
            total_trade_pnl = sum(t.net_amount + t.commission for t in completed_trades)
            avg_trade_return = total_trade_pnl / len(completed_trades) if completed_trades else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'total_trades': len(self.trades),
            'days_tracked': len(recent_snapshots)
        }
    
    def get_trade_ledger(self) -> pd.DataFrame:
        """Get trade ledger as DataFrame.
        
        Returns:
            DataFrame with all trades
        """
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = [asdict(trade) for trade in self.trades]
        df = pd.DataFrame(trades_data)
        
        # Convert order_type enum to string
        df['order_type'] = df['order_type'].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        df['reason'] = df['reason'].apply(lambda x: x.value if hasattr(x, 'value') else str(x))
        
        return df
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get current positions as DataFrame.
        
        Returns:
            DataFrame with position details
        """
        if not self.positions:
            return pd.DataFrame()
        
        positions_data = [asdict(pos) for pos in self.positions.values()]
        return pd.DataFrame(positions_data)
    
    def _save_portfolio_state(self) -> None:
        """Save portfolio state to disk."""
        try:
            state = {
                'cash': self.cash,
                'total_value': self.total_value,
                'last_rebalance_date': self.last_rebalance_date,
                'trade_counter': self.trade_counter,
                'positions': {ticker: asdict(pos) for ticker, pos in self.positions.items()},
                'trades': [asdict(trade) for trade in self.trades[-100:]],  # Keep last 100 trades
                'snapshots': [asdict(snap) for snap in self.snapshots[-30:]]  # Keep last 30 snapshots
            }
            
            # Convert enums to strings for JSON serialization
            for trade in state['trades']:
                if 'order_type' in trade:
                    trade['order_type'] = trade['order_type'].value if hasattr(trade['order_type'], 'value') else str(trade['order_type'])
                if 'reason' in trade:
                    trade['reason'] = trade['reason'].value if hasattr(trade['reason'], 'value') else str(trade['reason'])
            
            state_file = self.data_dir / "portfolio_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving portfolio state: {e}")
    
    def _load_portfolio_state(self) -> None:
        """Load portfolio state from disk."""
        try:
            state_file = self.data_dir / "portfolio_state.json"
            if not state_file.exists():
                return
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.cash = state.get('cash', self.initial_capital)
            self.total_value = state.get('total_value', self.initial_capital)
            self.last_rebalance_date = state.get('last_rebalance_date')
            self.trade_counter = state.get('trade_counter', 0)
            
            # Load positions
            positions_data = state.get('positions', {})
            for ticker, pos_data in positions_data.items():
                self.positions[ticker] = Position(**pos_data)
            
            # Load trades
            trades_data = state.get('trades', [])
            for trade_data in trades_data:
                # Convert string back to enum
                trade_data['order_type'] = OrderType(trade_data['order_type'])
                trade_data['reason'] = RebalanceReason(trade_data['reason'])
                self.trades.append(Trade(**trade_data))
            
            # Load snapshots
            snapshots_data = state.get('snapshots', [])
            for snap_data in snapshots_data:
                # Reconstruct positions in snapshot
                snap_positions = {}
                for ticker, pos_data in snap_data.get('positions', {}).items():
                    snap_positions[ticker] = Position(**pos_data)
                snap_data['positions'] = snap_positions
                self.snapshots.append(PortfolioSnapshot(**snap_data))
            
            logger.info("Portfolio state loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading portfolio state: {e}")


def main():
    """Main function for standalone testing."""
    import argparse
    import sys
    import os
    
    # Add parent directory for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from data_collector import MarketDataCollector
    from signals.signal_aggregator import SignalAggregator
    
    parser = argparse.ArgumentParser(description='Portfolio Manager')
    parser.add_argument('--status', action='store_true', help='Show portfolio status')
    parser.add_argument('--rebalance', action='store_true', help='Check and execute rebalancing')
    parser.add_argument('--performance', action='store_true', help='Show performance metrics')
    parser.add_argument('--trades', action='store_true', help='Show trade history')
    
    args = parser.parse_args()
    
    # Initialize portfolio manager
    portfolio = PortfolioManager()
    
    if args.status:
        # Show portfolio status
        print("\n" + "="*80)
        print("PORTFOLIO STATUS")
        print("="*80)
        
        print(f"\nTotal Value: ${portfolio.total_value:,.2f}")
        print(f"Cash: ${portfolio.cash:,.2f} ({portfolio.cash/portfolio.total_value:.1%})")
        print(f"Invested: ${portfolio.total_value - portfolio.cash:,.2f}")
        
        positions_df = portfolio.get_position_summary()
        if not positions_df.empty:
            print(f"\nPositions ({len(positions_df)}):")
            print("-" * 60)
            for _, pos in positions_df.iterrows():
                pnl_pct = pos['unrealized_pnl'] / (pos['shares'] * pos['avg_cost']) * 100 if pos['shares'] * pos['avg_cost'] > 0 else 0
                print(f"{pos['ticker']:>6} | {pos['shares']:>8.1f} shares | "
                      f"${pos['market_value']:>8,.0f} ({pos['weight']:>5.1%}) | "
                      f"P&L: {pnl_pct:>+6.1f}%")
    
    elif args.performance:
        # Show performance metrics
        metrics = portfolio.calculate_performance_metrics()
        
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        
        print(f"\nTotal Return: {metrics['total_return']:>+8.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:>+8.2%}")
        print(f"Volatility: {metrics['volatility']:>8.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:>8.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:>8.2%}")
        print(f"Win Rate: {metrics['win_rate']:>8.1%}")
        print(f"Total Trades: {metrics['total_trades']:>8.0f}")
        print(f"Days Tracked: {metrics['days_tracked']:>8.0f}")
    
    elif args.trades:
        # Show trade history
        trades_df = portfolio.get_trade_ledger()
        
        if not trades_df.empty:
            print("\n" + "="*80)
            print("TRADE HISTORY")
            print("="*80)
            
            for _, trade in trades_df.tail(10).iterrows():
                print(f"\n{trade['timestamp'][:10]} | {trade['trade_id']}")
                print(f"  {trade['order_type']} {trade['shares']} {trade['ticker']} @ ${trade['price']:.2f}")
                print(f"  Net: ${trade['net_amount']:+,.2f} | Commission: ${trade['commission']:.2f}")
                print(f"  Reason: {trade['reason']}")
        else:
            print("No trades found.")
    
    elif args.rebalance:
        # Check for rebalancing needs
        print("Checking rebalancing needs...")
        
        # Initialize data collector and signal aggregator
        data_collector = MarketDataCollector()
        signal_aggregator = SignalAggregator()
        
        # Get current market data
        tickers = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA', 'AMD']  # Example
        
        print("Generating signals...")
        signals_df = signal_aggregator.generate_aggregated_signals(tickers)
        
        if signals_df.empty:
            print("No signals generated. Cannot rebalance.")
            return
        
        # Get current prices
        price_data = {}
        for ticker in tickers:
            df = data_collector.load_from_parquet(ticker)
            if df is not None and not df.empty:
                price_data[ticker] = df.iloc[-1]['Close']
        
        if not price_data:
            print("No price data available. Cannot rebalance.")
            return
        
        # Update portfolio with current prices
        portfolio.update_prices(price_data)
        
        # Calculate target portfolio
        target_weights = portfolio.calculate_target_portfolio(signals_df, price_data)
        
        # Check if rebalancing is needed
        needs_rebalance, reasons = portfolio.check_rebalance_needed(target_weights)
        
        if needs_rebalance:
            print(f"\nRebalancing needed. Reasons:")
            for reason in reasons:
                print(f"  - {reason}")
            
            # Generate orders
            orders = portfolio.generate_rebalancing_orders(target_weights, price_data)
            
            print(f"\nGenerated {len(orders)} orders:")
            for order in orders:
                print(f"  {order['order_type'].value} {order['shares']} {order['ticker']} @ ${order['price']:.2f}")
            
            # Execute orders (in simulation)
            executed_trades = portfolio.execute_rebalancing_orders(orders)
            
            print(f"\nExecuted {len(executed_trades)} trades")
            
            # Take snapshot
            snapshot = portfolio.take_snapshot()
            print(f"Portfolio value: ${snapshot.total_value:,.2f}")
            
        else:
            print("No rebalancing needed.")
    
    else:
        print("Use --status, --performance, --trades, or --rebalance to interact with the portfolio")


if __name__ == "__main__":
    main()