#!/usr/bin/env python3
"""
Performance Tracking Module
Comprehensive performance analytics and signal attribution
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
import json
import warnings
warnings.filterwarnings('ignore')

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logging.warning("Matplotlib/Seaborn not available - plotting disabled")

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    date: str
    period: str  # daily, weekly, monthly
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    volatility: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    trades_count: int
    days_in_drawdown: int
    recovery_time: Optional[int]
    benchmark_return: float
    alpha: float
    beta: float


@dataclass
class SignalPerformance:
    """Container for signal-specific performance metrics."""
    signal_type: str
    date: str
    prediction_accuracy: float
    avg_return_when_bullish: float
    avg_return_when_bearish: float
    trades_generated: int
    successful_trades: int
    contribution_to_return: float
    sharpe_contribution: float


@dataclass
class TradeAttribution:
    """Container for trade attribution data."""
    trade_id: str
    ticker: str
    entry_date: str
    exit_date: Optional[str]
    entry_price: float
    exit_price: Optional[float]
    shares: float
    return_pct: float
    return_dollars: float
    signal_type: str
    signal_strength: float
    confidence_score: float
    holding_period: Optional[int]
    reason_for_exit: Optional[str]


class PerformanceTracker:
    """Comprehensive performance tracking and analytics system."""
    
    def __init__(self, db_path: str = "data/performance.db"):
        """Initialize performance tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Benchmark data (S&P 500 proxy)
        self.benchmark_returns = self._generate_benchmark_data()
        
        logger.info(f"Performance tracker initialized with database: {db_path}")
    
    def _init_database(self) -> None:
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Portfolio snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash REAL NOT NULL,
                invested_value REAL NOT NULL,
                daily_return REAL,
                cumulative_return REAL,
                num_positions INTEGER,
                largest_position_weight REAL,
                data_json TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                period TEXT NOT NULL,
                total_return REAL,
                sharpe_ratio REAL,
                sortino_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                win_rate REAL,
                profit_factor REAL,
                avg_win REAL,
                avg_loss REAL,
                trades_count INTEGER,
                days_in_drawdown INTEGER,
                recovery_time INTEGER,
                benchmark_return REAL,
                alpha REAL,
                beta REAL
            )
        ''')
        
        # Trade attribution table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trade_attribution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE NOT NULL,
                ticker TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                exit_date TEXT,
                entry_price REAL NOT NULL,
                exit_price REAL,
                shares REAL NOT NULL,
                return_pct REAL,
                return_dollars REAL,
                signal_type TEXT NOT NULL,
                signal_strength REAL,
                confidence_score REAL,
                holding_period INTEGER,
                reason_for_exit TEXT
            )
        ''')
        
        # Signal performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT NOT NULL,
                date TEXT NOT NULL,
                prediction_accuracy REAL,
                avg_return_when_bullish REAL,
                avg_return_when_bearish REAL,
                trades_generated INTEGER,
                successful_trades INTEGER,
                contribution_to_return REAL,
                sharpe_contribution REAL
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_portfolio_date ON portfolio_snapshots(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_date ON performance_metrics(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trade_attribution(ticker)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_signal ON trade_attribution(signal_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signal_perf_date ON signal_performance(date)')
        
        conn.commit()
        conn.close()
    
    def _generate_benchmark_data(self) -> pd.DataFrame:
        """Generate benchmark (S&P 500) returns for comparison."""
        # Generate synthetic S&P 500 data for demonstration
        dates = pd.date_range(start='2024-01-01', end='2025-12-31', freq='D')
        
        # Realistic S&P 500 parameters (10% annual return, 16% volatility)
        np.random.seed(42)  # For reproducible results
        daily_returns = np.random.normal(0.0004, 0.01, len(dates))  # ~10% annual, 16% vol
        
        # Add some correlation and trends
        for i in range(1, len(daily_returns)):
            daily_returns[i] += 0.05 * daily_returns[i-1]  # Small momentum effect
        
        benchmark_df = pd.DataFrame({
            'date': dates,
            'daily_return': daily_returns,
            'cumulative_return': (1 + pd.Series(daily_returns)).cumprod() - 1
        })
        
        return benchmark_df
    
    def record_portfolio_snapshot(self, 
                                portfolio_value: float,
                                cash: float,
                                positions: Dict[str, Any],
                                additional_data: Optional[Dict[str, Any]] = None) -> None:
        """Record a portfolio snapshot for performance tracking.
        
        Args:
            portfolio_value: Total portfolio value
            cash: Cash balance
            positions: Dictionary of positions
            additional_data: Additional data to store
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        date = datetime.now().date().isoformat()
        timestamp = datetime.now().isoformat()
        invested_value = portfolio_value - cash
        num_positions = len(positions)
        
        # Calculate largest position weight
        largest_position_weight = 0
        if positions and portfolio_value > 0:
            position_values = [pos.get('market_value', 0) for pos in positions.values()]
            largest_position_weight = max(position_values) / portfolio_value if position_values else 0
        
        # Get previous day's value for daily return calculation
        cursor.execute('''
            SELECT total_value FROM portfolio_snapshots 
            WHERE date < ? ORDER BY date DESC LIMIT 1
        ''', (date,))
        
        previous_value = cursor.fetchone()
        daily_return = None
        
        if previous_value and previous_value[0] > 0:
            daily_return = (portfolio_value / previous_value[0]) - 1
        
        # Calculate cumulative return (from first recorded value)
        cursor.execute('''
            SELECT total_value FROM portfolio_snapshots 
            ORDER BY date ASC LIMIT 1
        ''')
        
        first_value = cursor.fetchone()
        cumulative_return = None
        
        if first_value and first_value[0] > 0:
            cumulative_return = (portfolio_value / first_value[0]) - 1
        
        # Prepare additional data as JSON
        data_json = json.dumps(additional_data) if additional_data else None
        
        # Insert snapshot
        cursor.execute('''
            INSERT OR REPLACE INTO portfolio_snapshots 
            (date, timestamp, total_value, cash, invested_value, daily_return, 
             cumulative_return, num_positions, largest_position_weight, data_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (date, timestamp, portfolio_value, cash, invested_value, daily_return,
              cumulative_return, num_positions, largest_position_weight, data_json))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Recorded portfolio snapshot for {date}: ${portfolio_value:,.2f}")
    
    def record_trade(self,
                    trade_id: str,
                    ticker: str,
                    entry_date: str,
                    entry_price: float,
                    shares: float,
                    signal_type: str,
                    signal_strength: float,
                    confidence_score: float,
                    exit_date: Optional[str] = None,
                    exit_price: Optional[float] = None,
                    reason_for_exit: Optional[str] = None) -> None:
        """Record a trade for attribution analysis.
        
        Args:
            trade_id: Unique trade identifier
            ticker: Stock ticker
            entry_date: Trade entry date
            entry_price: Entry price
            shares: Number of shares
            signal_type: Type of signal that generated the trade
            signal_strength: Strength of the signal (0-1)
            confidence_score: Confidence in the trade (0-1)
            exit_date: Exit date (if trade is closed)
            exit_price: Exit price (if trade is closed)
            reason_for_exit: Reason for exit
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate returns if trade is closed
        return_pct = None
        return_dollars = None
        holding_period = None
        
        if exit_date and exit_price:
            return_pct = (exit_price / entry_price) - 1
            return_dollars = (exit_price - entry_price) * shares
            
            entry_dt = datetime.fromisoformat(entry_date)
            exit_dt = datetime.fromisoformat(exit_date)
            holding_period = (exit_dt - entry_dt).days
        
        cursor.execute('''
            INSERT OR REPLACE INTO trade_attribution
            (trade_id, ticker, entry_date, exit_date, entry_price, exit_price,
             shares, return_pct, return_dollars, signal_type, signal_strength,
             confidence_score, holding_period, reason_for_exit)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (trade_id, ticker, entry_date, exit_date, entry_price, exit_price,
              shares, return_pct, return_dollars, signal_type, signal_strength,
              confidence_score, holding_period, reason_for_exit))
        
        conn.commit()
        conn.close()
        
        logger.debug(f"Recorded trade: {trade_id} - {ticker}")
    
    def calculate_performance_metrics(self, 
                                    start_date: Optional[str] = None,
                                    end_date: Optional[str] = None,
                                    period: str = "daily") -> PerformanceMetrics:
        """Calculate comprehensive performance metrics.
        
        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            period: Period for metrics (daily, weekly, monthly)
            
        Returns:
            PerformanceMetrics object
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query with date filters
        query = "SELECT * FROM portfolio_snapshots WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty or len(df) < 2:
            # Return default metrics if insufficient data
            return PerformanceMetrics(
                date=datetime.now().date().isoformat(),
                period=period,
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
        
        # Calculate returns
        df['daily_return'] = df['total_value'].pct_change()
        df = df.dropna()
        
        if df.empty:
            return PerformanceMetrics(
                date=datetime.now().date().isoformat(),
                period=period,
                total_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, volatility=0.0, win_rate=0.0,
                profit_factor=0.0, avg_win=0.0, avg_loss=0.0, trades_count=0,
                days_in_drawdown=0, recovery_time=None, benchmark_return=0.0,
                alpha=0.0, beta=0.0
            )
        
        returns = df['daily_return']
        
        # Total return
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        avg_return = returns.mean() * 252
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 1 else 0
        sortino_ratio = (avg_return - risk_free_rate) / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Days in drawdown and recovery time
        days_in_drawdown = (drawdown < -0.01).sum()  # Days with >1% drawdown
        
        # Find recovery time (days from max drawdown to recovery)
        recovery_time = None
        max_dd_idx = drawdown.idxmin()
        if max_dd_idx in drawdown.index:
            recovery_idx = drawdown[max_dd_idx:].ge(-0.001).idxmax()  # Within 0.1% of peak
            if recovery_idx and recovery_idx != max_dd_idx:
                recovery_time = (recovery_idx - max_dd_idx)
        
        # Win rate and profit factor
        winning_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')
        
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        
        # Get benchmark return for same period
        benchmark_data = self.benchmark_returns[
            (self.benchmark_returns['date'] >= df['date'].iloc[0]) &
            (self.benchmark_returns['date'] <= df['date'].iloc[-1])
        ]
        
        benchmark_return = 0
        alpha = 0
        beta = 0
        
        if not benchmark_data.empty:
            benchmark_return = benchmark_data['daily_return'].sum()
            
            # Calculate alpha and beta
            if len(benchmark_data) > 10 and len(returns) > 10:
                # Align dates
                portfolio_aligned = df.set_index('date')['daily_return']
                benchmark_aligned = benchmark_data.set_index('date')['daily_return']
                
                # Find common dates
                common_dates = portfolio_aligned.index.intersection(benchmark_aligned.index)
                
                if len(common_dates) > 10:
                    port_returns = portfolio_aligned.loc[common_dates]
                    bench_returns = benchmark_aligned.loc[common_dates]
                    
                    # Calculate beta (covariance / variance)
                    covariance = np.cov(port_returns, bench_returns)[0, 1]
                    benchmark_variance = np.var(bench_returns)
                    
                    if benchmark_variance > 0:
                        beta = covariance / benchmark_variance
                        
                        # Calculate alpha
                        alpha = (port_returns.mean() - risk_free_rate/252) - beta * (bench_returns.mean() - risk_free_rate/252)
                        alpha *= 252  # Annualize
        
        # Get trade count
        trade_conn = sqlite3.connect(self.db_path)
        trade_query = "SELECT COUNT(*) FROM trade_attribution WHERE entry_date >= ? AND entry_date <= ?"
        trade_cursor = trade_conn.cursor()
        trade_cursor.execute(trade_query, (df['date'].iloc[0], df['date'].iloc[-1]))
        trades_count = trade_cursor.fetchone()[0]
        trade_conn.close()
        
        return PerformanceMetrics(
            date=end_date or datetime.now().date().isoformat(),
            period=period,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            volatility=volatility,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades_count=trades_count,
            days_in_drawdown=days_in_drawdown,
            recovery_time=recovery_time,
            benchmark_return=benchmark_return,
            alpha=alpha,
            beta=beta
        )
    
    def analyze_signal_performance(self, 
                                 signal_type: str,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> SignalPerformance:
        """Analyze performance of a specific signal type.
        
        Args:
            signal_type: Type of signal to analyze
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            SignalPerformance object
        """
        conn = sqlite3.connect(self.db_path)
        
        # Build query for trades with this signal type
        query = '''
            SELECT * FROM trade_attribution 
            WHERE signal_type = ? AND exit_date IS NOT NULL
        '''
        params = [signal_type]
        
        if start_date:
            query += " AND entry_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND entry_date <= ?"
            params.append(end_date)
        
        trades_df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if trades_df.empty:
            return SignalPerformance(
                signal_type=signal_type,
                date=datetime.now().date().isoformat(),
                prediction_accuracy=0.0,
                avg_return_when_bullish=0.0,
                avg_return_when_bearish=0.0,
                trades_generated=0,
                successful_trades=0,
                contribution_to_return=0.0,
                sharpe_contribution=0.0
            )
        
        # Calculate metrics
        total_trades = len(trades_df)
        successful_trades = (trades_df['return_pct'] > 0).sum()
        prediction_accuracy = successful_trades / total_trades if total_trades > 0 else 0
        
        # Separate bullish vs bearish signals (assuming signal_strength > 0.5 is bullish)
        bullish_trades = trades_df[trades_df['signal_strength'] > 0.5]
        bearish_trades = trades_df[trades_df['signal_strength'] <= 0.5]
        
        avg_return_when_bullish = bullish_trades['return_pct'].mean() if not bullish_trades.empty else 0
        avg_return_when_bearish = bearish_trades['return_pct'].mean() if not bearish_trades.empty else 0
        
        # Total contribution to portfolio returns
        contribution_to_return = trades_df['return_dollars'].sum()
        
        # Sharpe contribution (simplified)
        returns = trades_df['return_pct']
        sharpe_contribution = returns.mean() / returns.std() if len(returns) > 1 and returns.std() > 0 else 0
        
        return SignalPerformance(
            signal_type=signal_type,
            date=end_date or datetime.now().date().isoformat(),
            prediction_accuracy=prediction_accuracy,
            avg_return_when_bullish=avg_return_when_bullish,
            avg_return_when_bearish=avg_return_when_bearish,
            trades_generated=total_trades,
            successful_trades=successful_trades,
            contribution_to_return=contribution_to_return,
            sharpe_contribution=sharpe_contribution
        )
    
    def generate_weekly_report(self, week_ending: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive weekly performance report.
        
        Args:
            week_ending: End date of the week to analyze
            
        Returns:
            Dictionary containing report data
        """
        if week_ending is None:
            week_ending = datetime.now().date().isoformat()
        
        week_start = (datetime.fromisoformat(week_ending) - timedelta(days=7)).date().isoformat()
        
        # Calculate weekly metrics
        weekly_metrics = self.calculate_performance_metrics(
            start_date=week_start,
            end_date=week_ending,
            period="weekly"
        )
        
        # Get signal performance for all signal types
        conn = sqlite3.connect(self.db_path)
        signal_types_query = "SELECT DISTINCT signal_type FROM trade_attribution"
        signal_types_df = pd.read_sql_query(signal_types_query, conn)
        conn.close()
        
        signal_performances = {}
        for signal_type in signal_types_df['signal_type']:
            signal_perf = self.analyze_signal_performance(
                signal_type, week_start, week_ending
            )
            signal_performances[signal_type] = asdict(signal_perf)
        
        # Get top performers and underperformers
        top_performers = []
        underperformers = []
        
        conn = sqlite3.connect(self.db_path)
        trades_query = '''
            SELECT ticker, return_pct, signal_type, confidence_score
            FROM trade_attribution 
            WHERE entry_date >= ? AND entry_date <= ?
            AND exit_date IS NOT NULL
            ORDER BY return_pct DESC
        '''
        trades_df = pd.read_sql_query(trades_query, conn, params=[week_start, week_ending])
        conn.close()
        
        if not trades_df.empty:
            top_performers = trades_df.head(5).to_dict('records')
            underperformers = trades_df.tail(5).to_dict('records')
        
        # Portfolio composition analysis
        conn = sqlite3.connect(self.db_path)
        latest_snapshot_query = '''
            SELECT * FROM portfolio_snapshots 
            WHERE date <= ? ORDER BY date DESC LIMIT 1
        '''
        latest_snapshot_df = pd.read_sql_query(latest_snapshot_query, conn, params=[week_ending])
        conn.close()
        
        portfolio_composition = {}
        if not latest_snapshot_df.empty:
            portfolio_composition = {
                'total_value': latest_snapshot_df['total_value'].iloc[0],
                'cash_percentage': (latest_snapshot_df['cash'].iloc[0] / latest_snapshot_df['total_value'].iloc[0]) * 100,
                'num_positions': latest_snapshot_df['num_positions'].iloc[0],
                'largest_position_weight': latest_snapshot_df['largest_position_weight'].iloc[0] * 100
            }
        
        # Strategy recommendations
        recommendations = self.generate_strategy_recommendations(signal_performances, weekly_metrics)
        
        report = {
            'report_date': week_ending,
            'period': f"{week_start} to {week_ending}",
            'performance_metrics': asdict(weekly_metrics),
            'signal_performances': signal_performances,
            'top_performers': top_performers,
            'underperformers': underperformers,
            'portfolio_composition': portfolio_composition,
            'strategy_recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def generate_strategy_recommendations(self, 
                                        signal_performances: Dict[str, Dict],
                                        portfolio_metrics: PerformanceMetrics) -> List[Dict[str, str]]:
        """Generate strategy improvement recommendations based on performance data.
        
        Args:
            signal_performances: Dictionary of signal performance data
            portfolio_metrics: Portfolio performance metrics
            
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Analyze signal effectiveness
        if signal_performances:
            # Find best performing signal
            best_signal = max(signal_performances.items(), 
                            key=lambda x: x[1]['prediction_accuracy'])
            
            # Find worst performing signal
            worst_signal = min(signal_performances.items(),
                             key=lambda x: x[1]['prediction_accuracy'])
            
            if best_signal[1]['prediction_accuracy'] > 0.6:
                recommendations.append({
                    'type': 'Signal Weighting',
                    'priority': 'HIGH',
                    'recommendation': f"Increase weight for {best_signal[0]} signal (accuracy: {best_signal[1]['prediction_accuracy']:.1%})",
                    'impact': 'Positive'
                })
            
            if worst_signal[1]['prediction_accuracy'] < 0.4:
                recommendations.append({
                    'type': 'Signal Filtering',
                    'priority': 'MEDIUM',
                    'recommendation': f"Review or reduce weight for {worst_signal[0]} signal (accuracy: {worst_signal[1]['prediction_accuracy']:.1%})",
                    'impact': 'Risk Reduction'
                })
        
        # Portfolio risk recommendations
        if portfolio_metrics.max_drawdown < -0.15:  # > 15% drawdown
            recommendations.append({
                'type': 'Risk Management',
                'priority': 'HIGH',
                'recommendation': f"Reduce position sizes or tighten stop-losses. Current max drawdown: {portfolio_metrics.max_drawdown:.1%}",
                'impact': 'Risk Reduction'
            })
        
        if portfolio_metrics.sharpe_ratio < 0.5:
            recommendations.append({
                'type': 'Return Enhancement',
                'priority': 'MEDIUM',
                'recommendation': f"Low risk-adjusted returns (Sharpe: {portfolio_metrics.sharpe_ratio:.2f}). Consider more selective trade entry.",
                'impact': 'Return Enhancement'
            })
        
        # Diversification recommendations
        if portfolio_metrics.volatility > 0.25:  # > 25% volatility
            recommendations.append({
                'type': 'Diversification',
                'priority': 'MEDIUM',
                'recommendation': f"High portfolio volatility ({portfolio_metrics.volatility:.1%}). Consider increasing diversification.",
                'impact': 'Risk Reduction'
            })
        
        # Win rate recommendations
        if portfolio_metrics.win_rate < 0.45:  # < 45% win rate
            recommendations.append({
                'type': 'Trade Selection',
                'priority': 'MEDIUM',
                'recommendation': f"Low win rate ({portfolio_metrics.win_rate:.1%}). Review entry criteria and signal thresholds.",
                'impact': 'Accuracy Improvement'
            })
        
        # Benchmark comparison
        if portfolio_metrics.alpha < 0:
            recommendations.append({
                'type': 'Alpha Generation',
                'priority': 'HIGH',
                'recommendation': f"Negative alpha ({portfolio_metrics.alpha:.3f}). Strategy underperforming risk-adjusted benchmark.",
                'impact': 'Return Enhancement'
            })
        
        return recommendations
    
    def save_metrics_to_db(self, metrics: PerformanceMetrics) -> None:
        """Save performance metrics to database.
        
        Args:
            metrics: PerformanceMetrics object to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO performance_metrics
            (date, period, total_return, sharpe_ratio, sortino_ratio, max_drawdown,
             volatility, win_rate, profit_factor, avg_win, avg_loss, trades_count,
             days_in_drawdown, recovery_time, benchmark_return, alpha, beta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            metrics.date, metrics.period, metrics.total_return, metrics.sharpe_ratio,
            metrics.sortino_ratio, metrics.max_drawdown, metrics.volatility,
            metrics.win_rate, metrics.profit_factor, metrics.avg_win, metrics.avg_loss,
            metrics.trades_count, metrics.days_in_drawdown, metrics.recovery_time,
            metrics.benchmark_return, metrics.alpha, metrics.beta
        ))
        
        conn.commit()
        conn.close()
    
    def get_rolling_metrics(self, window_days: int = 90) -> pd.DataFrame:
        """Get rolling performance metrics.
        
        Args:
            window_days: Rolling window size in days
            
        Returns:
            DataFrame with rolling metrics
        """
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT date, total_value, daily_return 
            FROM portfolio_snapshots 
            ORDER BY date
        ''', conn)
        conn.close()
        
        if df.empty or len(df) < window_days:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # Calculate rolling metrics
        rolling_metrics = pd.DataFrame(index=df.index)
        
        # Rolling returns
        rolling_metrics['rolling_return'] = df['total_value'].pct_change(window_days)
        
        # Rolling Sharpe ratio
        rolling_returns = df['daily_return'].rolling(window_days)
        rolling_metrics['rolling_sharpe'] = (
            rolling_returns.mean() * 252 - 0.02
        ) / (rolling_returns.std() * np.sqrt(252))
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = rolling_returns.std() * np.sqrt(252)
        
        # Rolling max drawdown
        rolling_cumulative = (1 + df['daily_return']).rolling(window_days).apply(lambda x: x.prod())
        rolling_max = rolling_cumulative.rolling(window_days).max()
        rolling_metrics['rolling_max_drawdown'] = ((rolling_cumulative - rolling_max) / rolling_max).rolling(window_days).min()
        
        return rolling_metrics.dropna()


def main():
    """Test the performance tracker."""
    import random
    
    # Initialize tracker
    tracker = PerformanceTracker("data/test_performance.db")
    
    print("Testing Performance Tracker...")
    
    # Generate some sample data
    start_date = datetime(2024, 1, 1)
    current_value = 10000
    
    for i in range(100):  # 100 days of data
        date = start_date + timedelta(days=i)
        
        # Simulate portfolio value changes
        daily_return = random.normalvariate(0.0005, 0.015)  # ~13% annual return, 24% volatility
        current_value *= (1 + daily_return)
        
        # Record portfolio snapshot
        positions = {
            'AAPL': {'market_value': current_value * 0.2},
            'MSFT': {'market_value': current_value * 0.15},
            'GOOGL': {'market_value': current_value * 0.1}
        }
        
        tracker.record_portfolio_snapshot(
            portfolio_value=current_value,
            cash=current_value * 0.1,
            positions=positions
        )
        
        # Record some trades
        if i % 10 == 0:  # Every 10 days
            trade_id = f"TRADE_{date.strftime('%Y%m%d')}_{i}"
            ticker = random.choice(['AAPL', 'MSFT', 'GOOGL', 'NVDA'])
            entry_price = random.uniform(100, 300)
            shares = random.uniform(10, 100)
            signal_type = random.choice(['LLM_Sentiment', 'Technical_Momentum', 'Sector_Rotation'])
            
            tracker.record_trade(
                trade_id=trade_id,
                ticker=ticker,
                entry_date=date.isoformat(),
                entry_price=entry_price,
                shares=shares,
                signal_type=signal_type,
                signal_strength=random.uniform(0.3, 0.9),
                confidence_score=random.uniform(0.4, 0.8)
            )
            
            # Close some trades
            if i > 20:
                exit_price = entry_price * random.uniform(0.95, 1.05)
                tracker.record_trade(
                    trade_id=trade_id,
                    ticker=ticker,
                    entry_date=date.isoformat(),
                    entry_price=entry_price,
                    shares=shares,
                    signal_type=signal_type,
                    signal_strength=random.uniform(0.3, 0.9),
                    confidence_score=random.uniform(0.4, 0.8),
                    exit_date=(date + timedelta(days=random.randint(1, 10))).isoformat(),
                    exit_price=exit_price,
                    reason_for_exit="Target/Stop"
                )
    
    # Calculate and display metrics
    print("\nCalculating performance metrics...")
    metrics = tracker.calculate_performance_metrics()
    
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    print(f"Win Rate: {metrics.win_rate:.1%}")
    print(f"Volatility: {metrics.volatility:.1%}")
    
    # Analyze signal performance
    print("\nSignal Performance Analysis:")
    for signal_type in ['LLM_Sentiment', 'Technical_Momentum', 'Sector_Rotation']:
        signal_perf = tracker.analyze_signal_performance(signal_type)
        print(f"{signal_type}: {signal_perf.prediction_accuracy:.1%} accuracy, "
              f"{signal_perf.trades_generated} trades")
    
    # Generate weekly report
    print("\nGenerating weekly report...")
    report = tracker.generate_weekly_report()
    
    print(f"Report Period: {report['period']}")
    print(f"Weekly Return: {report['performance_metrics']['total_return']:.2%}")
    print(f"Number of Recommendations: {len(report['strategy_recommendations'])}")
    
    for rec in report['strategy_recommendations']:
        print(f"- {rec['type']}: {rec['recommendation']}")
    
    print("\nPerformance tracker test completed!")


if __name__ == "__main__":
    main()