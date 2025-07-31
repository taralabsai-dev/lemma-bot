#!/usr/bin/env python3
"""
Autonomous Trading System - Main Orchestration Script
Runs the complete trading pipeline on a scheduled basis
"""

import os
import sys
import time
import logging
import argparse
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import signal
import threading
from contextlib import contextmanager

# Third-party imports
import schedule
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Local imports
from portfolio_manager import PortfolioManager, RebalanceReason
from risk_manager import RiskManager, RiskLevel
from data_collector import MarketDataCollector
from signals.signal_aggregator import SignalAggregator
from llm_analyst import LLMAnalyst
from news_collector import NewsCollector
from trade_approval_system import TradeApprovalSystem, TradeAction

# Configure logging
def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up comprehensive logging system."""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger('autonomous_trading')
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = log_dir / f"trading_system_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Global logger
logger = setup_logging()

class PipelineStage:
    """Represents a stage in the trading pipeline."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.success: bool = False
        self.error_message: Optional[str] = None
        self.result_data: Optional[Dict[str, Any]] = None
    
    def start(self):
        """Mark stage as started."""
        self.start_time = datetime.now()
        logger.info(f"🚀 Starting stage: {self.name} - {self.description}")
    
    def complete(self, result_data: Optional[Dict[str, Any]] = None):
        """Mark stage as completed successfully."""
        self.end_time = datetime.now()
        self.success = True
        self.result_data = result_data
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"✅ Completed stage: {self.name} ({duration:.1f}s)")
    
    def fail(self, error_message: str):
        """Mark stage as failed."""
        self.end_time = datetime.now()
        self.success = False
        self.error_message = error_message
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        logger.error(f"❌ Failed stage: {self.name} ({duration:.1f}s) - {error_message}")
    
    @property
    def duration(self) -> float:
        """Get stage duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


class TradingSystemOrchestrator:
    """Main orchestrator for the autonomous trading system."""
    
    def __init__(self, dry_run: bool = False):
        """Initialize the orchestrator.
        
        Args:
            dry_run: If True, don't execute actual trades
        """
        self.dry_run = dry_run
        self.running = False
        self.last_run_time: Optional[datetime] = None
        self.pipeline_stages: List[PipelineStage] = []
        
        # Email configuration
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email_user': os.getenv('EMAIL_USER', ''),
            'email_password': os.getenv('EMAIL_PASSWORD', ''),
            'notification_emails': [email.strip() for email in os.getenv('NOTIFICATION_EMAILS', '').split(',') if email.strip()]
        }
        
        # Components (initialized on demand)
        self.portfolio_manager: Optional[PortfolioManager] = None
        self.risk_manager: Optional[RiskManager] = None
        self.data_collector: Optional[MarketDataCollector] = None
        self.signal_aggregator: Optional[SignalAggregator] = None
        self.llm_analyst: Optional[LLMAnalyst] = None
        self.news_collector: Optional[NewsCollector] = None
        self.trade_approval_system: Optional[TradeApprovalSystem] = None
        
        # Pipeline configuration
        self.max_positions = 8
        self.rebalance_threshold = 0.20  # 20% drift threshold
        
        logger.info(f"🤖 Autonomous Trading System Orchestrator initialized {'(DRY RUN MODE)' if dry_run else ''}")
    
    def initialize_components(self) -> bool:
        """Initialize all trading system components."""
        stage = PipelineStage("INIT", "Initialize system components")
        stage.start()
        
        try:
            logger.info("Initializing trading system components...")
            
            self.portfolio_manager = PortfolioManager()
            self.risk_manager = RiskManager()
            self.data_collector = MarketDataCollector()
            self.signal_aggregator = SignalAggregator()
            self.llm_analyst = LLMAnalyst()
            self.news_collector = NewsCollector()
            self.trade_approval_system = TradeApprovalSystem()
            
            stage.complete({
                'components_initialized': 7,
                'dry_run_mode': self.dry_run
            })
            return True
            
        except Exception as e:
            stage.fail(f"Component initialization failed: {str(e)}")
            return False
        finally:
            self.pipeline_stages.append(stage)
    
    def fetch_market_data(self) -> bool:
        """Fetch latest market data for all stocks."""
        stage = PipelineStage("DATA", "Fetch market data")
        stage.start()
        
        try:
            logger.info("Fetching latest market data...")
            
            # Get stock universe
            tickers = MarketDataCollector.TECH_UNIVERSE
            
            # Update market data
            results = self.data_collector.update_all_stocks(tickers, force_full_download=False)
            
            successful_updates = sum(results.values())
            failed_updates = len(results) - successful_updates
            
            if successful_updates < len(tickers) * 0.8:  # Less than 80% success
                stage.fail(f"Market data update failed for too many stocks: {failed_updates}/{len(tickers)} failed")
                return False
            
            stage.complete({
                'stocks_updated': successful_updates,
                'stocks_failed': failed_updates,
                'total_stocks': len(tickers)
            })
            return True
            
        except Exception as e:
            stage.fail(f"Market data fetch failed: {str(e)}")
            return False
        finally:
            self.pipeline_stages.append(stage)
    
    def collect_news_data(self) -> bool:
        """Collect news data for analysis."""
        stage = PipelineStage("NEWS", "Collect news data")
        stage.start()
        
        try:
            logger.info("Collecting news data...")
            
            # Get top stocks for news collection
            tickers = MarketDataCollector.TECH_UNIVERSE[:15]  # Top 15 stocks
            
            # Collect news
            news_results = self.news_collector.collect_multiple_stocks(tickers, use_cache=True)
            
            total_headlines = sum(len(headlines) for headlines in news_results.values())
            stocks_with_news = len([t for t, news in news_results.items() if news])
            
            stage.complete({
                'stocks_processed': len(news_results),
                'stocks_with_news': stocks_with_news,
                'total_headlines': total_headlines
            })
            return True
            
        except Exception as e:
            stage.fail(f"News collection failed: {str(e)}")
            return False
        finally:
            self.pipeline_stages.append(stage)
    
    def run_llm_analysis(self) -> Dict[str, Any]:
        """Run LLM analysis on market data and news."""
        stage = PipelineStage("LLM", "Run LLM analysis")
        stage.start()
        
        try:
            logger.info("Running LLM analysis...")
            
            # Get market data
            stock_data = {}
            tickers = MarketDataCollector.TECH_UNIVERSE[:15]  # Top 15 for analysis
            
            for ticker in tickers:
                df = self.data_collector.load_from_parquet(ticker)
                if df is not None and not df.empty:
                    stock_data[ticker] = df
            
            if not stock_data:
                stage.fail("No market data available for LLM analysis")
                return {}
            
            # Generate market regime analysis
            market_summary = self.llm_analyst.get_market_summary(stock_data)
            regime_analysis = self.llm_analyst.analyze_market_regime(market_summary)
            
            # Analyze individual stocks
            analysis_results = {}
            for ticker in list(stock_data.keys())[:10]:  # Limit to 10 for efficiency
                try:
                    # Get news for this ticker
                    news = self.news_collector.get_weekly_news(ticker, weeks_back=1)
                    if not news:
                        news = []
                    
                    # Analyze sentiment
                    if news:
                        sentiment_analysis = self.llm_analyst.analyze_news_sentiment(ticker, news[:10])
                        analysis_results[ticker] = sentiment_analysis
                    
                except Exception as e:
                    logger.warning(f"LLM analysis failed for {ticker}: {e}")
                    continue
            
            stage.complete({
                'market_regime': regime_analysis.get('regime', 'Unknown'),
                'regime_confidence': regime_analysis.get('confidence', 0),
                'stocks_analyzed': len(analysis_results),
                'analysis_results': analysis_results
            })
            
            return {
                'regime_analysis': regime_analysis,
                'stock_analyses': analysis_results
            }
            
        except Exception as e:
            stage.fail(f"LLM analysis failed: {str(e)}")
            return {}
        finally:
            self.pipeline_stages.append(stage)
    
    def generate_trading_signals(self, llm_results: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Generate all trading signals."""
        stage = PipelineStage("SIGNALS", "Generate trading signals")
        stage.start()
        
        try:
            logger.info("Generating trading signals...")
            
            # Get tickers for signal generation
            tickers = MarketDataCollector.TECH_UNIVERSE[:15]
            
            # Generate aggregated signals
            signals_df = self.signal_aggregator.generate_aggregated_signals(
                tickers, 
                use_regime_adjustment=True
            )
            
            if signals_df.empty:
                stage.fail("No trading signals generated")
                return None
            
            # Select top candidates
            top_candidates = self.signal_aggregator.select_top_candidates(
                signals_df, 
                top_n=self.max_positions,
                min_score=0.5
            )
            
            signals_list = signals_df.to_dict('records')
            
            stage.complete({
                'total_signals': len(signals_df),
                'top_candidates': len(top_candidates),
                'avg_signal_strength': signals_df['aggregate_score'].mean(),
                'max_signal_strength': signals_df['aggregate_score'].max()
            })
            
            return signals_list
            
        except Exception as e:
            stage.fail(f"Signal generation failed: {str(e)}")
            return None
        finally:
            self.pipeline_stages.append(stage)
    
    def create_rebalancing_trades(self, signals: List[Dict[str, Any]]) -> bool:
        """Create rebalancing trades based on signals."""
        stage = PipelineStage("REBALANCE", "Create rebalancing trades")
        stage.start()
        
        try:
            logger.info("Creating rebalancing trades...")
            
            if not signals:
                stage.fail("No signals available for rebalancing")
                return False
            
            # Convert signals to DataFrame
            signals_df = pd.DataFrame(signals)
            
            # Get current prices
            price_data = {}
            for _, signal in signals_df.iterrows():
                ticker = signal['ticker']
                df = self.data_collector.load_from_parquet(ticker)
                if df is not None and not df.empty:
                    price_data[ticker] = df.iloc[-1]['Close']
            
            if not price_data:
                stage.fail("No price data available for rebalancing")
                return False
            
            # Update portfolio with current prices
            self.portfolio_manager.update_prices(price_data)
            
            # Calculate target portfolio
            target_weights = self.portfolio_manager.calculate_target_portfolio(
                signals_df, 
                price_data, 
                max_positions=self.max_positions
            )
            
            # Check if rebalancing is needed
            needs_rebalance, reasons = self.portfolio_manager.check_rebalance_needed(
                target_weights, 
                drift_threshold=self.rebalance_threshold
            )
            
            if not needs_rebalance:
                stage.complete({
                    'rebalance_needed': False,
                    'reason': 'Portfolio within acceptable drift'
                })
                return True
            
            # Generate rebalancing orders
            orders = self.portfolio_manager.generate_rebalancing_orders(
                target_weights, 
                price_data, 
                reason=RebalanceReason.WEEKLY
            )
            
            if not orders:
                stage.complete({
                    'rebalance_needed': True,
                    'orders_generated': 0
                })
                return True
            
            # Create pending trades for approval
            trades_created = 0
            for order in orders:
                try:
                    # Get signal data for this ticker
                    signal_data = signals_df[signals_df['ticker'] == order['ticker']]
                    
                    if signal_data.empty:
                        continue
                    
                    signal_row = signal_data.iloc[0]
                    
                    # Create pending trade
                    self.trade_approval_system.create_pending_trade(
                        ticker=order['ticker'],
                        action=TradeAction.BUY if order['order_type'].value == 'BUY' else TradeAction.SELL,
                        shares=order['shares'],
                        price=order['price'],
                        reasoning=signal_row.get('reasoning', 'Rebalancing trade'),
                        llm_analysis=f"Signal strength: {signal_row.get('aggregate_score', 0):.3f}",
                        confidence_score=signal_row.get('llm_sentiment_signal', 0.5),
                        signal_strength=signal_row.get('aggregate_score', 0.5),
                        portfolio_impact={
                            'weight_change': target_weights.get(order['ticker'], 0),
                            'risk_description': 'Systematic rebalancing'
                        }
                    )
                    trades_created += 1
                    
                except Exception as e:
                    logger.error(f"Error creating pending trade for {order['ticker']}: {e}")
                    continue
            
            stage.complete({
                'rebalance_needed': True,
                'orders_generated': len(orders),
                'trades_created': trades_created,
                'reasons': reasons
            })
            
            # Send notification if trades were created
            if trades_created > 0 and not self.dry_run:
                self.trade_approval_system.notify_pending_trades()
            
            return True
            
        except Exception as e:
            stage.fail(f"Rebalancing failed: {str(e)}")
            return False
        finally:
            self.pipeline_stages.append(stage)
    
    def update_risk_metrics(self) -> bool:
        """Update risk metrics and check for violations."""
        stage = PipelineStage("RISK", "Update risk metrics")
        stage.start()
        
        try:
            logger.info("Updating risk metrics...")
            
            # Get current portfolio state
            positions = {}
            for ticker, position in self.portfolio_manager.positions.items():
                positions[ticker] = {
                    'market_value': position.market_value,
                    'weight': position.weight,
                    'shares': position.shares
                }
            
            # Get returns data for risk calculations
            returns_data = {}
            for ticker in positions.keys():
                df = self.data_collector.load_from_parquet(ticker)
                if df is not None and len(df) > 20:
                    returns_data[ticker] = df['Daily_Return'].dropna()
            
            # Calculate portfolio metrics
            portfolio_metrics = self.portfolio_manager.calculate_performance_metrics()
            max_drawdown = portfolio_metrics.get('max_drawdown', 0)
            
            # Update risk metrics
            risk_metrics = self.risk_manager.update_risk_metrics(
                portfolio_value=self.portfolio_manager.total_value,
                positions=positions,
                returns_data=returns_data,
                max_drawdown=max_drawdown
            )
            
            # Get active risk alerts
            active_alerts = self.risk_manager.get_active_alerts()
            critical_alerts = self.risk_manager.get_active_alerts(RiskLevel.CRITICAL)
            
            stage.complete({
                'risk_score': risk_metrics.risk_score,
                'max_drawdown': max_drawdown,
                'portfolio_volatility': risk_metrics.portfolio_volatility,
                'active_alerts': len(active_alerts),
                'critical_alerts': len(critical_alerts),
                'risk_off_mode': self.risk_manager.risk_off_mode
            })
            
            return True
            
        except Exception as e:
            stage.fail(f"Risk metrics update failed: {str(e)}")
            return False
        finally:
            self.pipeline_stages.append(stage)
    
    def run_full_pipeline(self) -> bool:
        """Run the complete trading pipeline."""
        pipeline_start = datetime.now()
        logger.info("="*80)
        logger.info(f"🚀 STARTING AUTONOMOUS TRADING PIPELINE {'(DRY RUN)' if self.dry_run else ''}")
        logger.info(f"📅 Start time: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)
        
        self.pipeline_stages.clear()
        success = True
        
        try:
            # Stage 1: Initialize components
            if not self.initialize_components():
                return False
            
            # Stage 2: Fetch market data
            if not self.fetch_market_data():
                return False
            
            # Stage 3: Collect news
            if not self.collect_news_data():
                return False
            
            # Stage 4: Run LLM analysis
            llm_results = self.run_llm_analysis()
            if not llm_results:
                return False
            
            # Stage 5: Generate signals
            signals = self.generate_trading_signals(llm_results)
            if signals is None:
                return False
            
            # Stage 6: Create rebalancing trades
            if not self.create_rebalancing_trades(signals):
                return False
            
            # Stage 7: Update risk metrics
            if not self.update_risk_metrics():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"💥 PIPELINE FAILED: {str(e)}")
            logger.error(traceback.format_exc())
            return False
            
        finally:
            pipeline_end = datetime.now()
            duration = (pipeline_end - pipeline_start).total_seconds()
            
            # Generate pipeline summary
            self.generate_pipeline_summary(pipeline_start, pipeline_end, success)
            
            # Send notification
            if not success:
                self.send_error_notification()
            
            self.last_run_time = pipeline_end
    
    def generate_pipeline_summary(self, start_time: datetime, end_time: datetime, success: bool) -> None:
        """Generate and log pipeline execution summary."""
        duration = (end_time - start_time).total_seconds()
        
        logger.info("="*80)
        logger.info(f"📊 PIPELINE SUMMARY")
        logger.info("="*80)
        logger.info(f"🕐 Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"🕐 End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"⏱️  Duration: {duration:.1f} seconds")
        logger.info(f"✅ Status: {'SUCCESS' if success else 'FAILED'}")
        logger.info("")
        
        # Stage-by-stage summary
        logger.info("📋 Stage Summary:")
        for stage in self.pipeline_stages:
            status = "✅" if stage.success else "❌"
            logger.info(f"  {status} {stage.name}: {stage.description} ({stage.duration:.1f}s)")
            if not stage.success and stage.error_message:
                logger.info(f"     Error: {stage.error_message}")
        
        logger.info("="*80)
        
        # Save summary to file
        self.save_pipeline_summary(start_time, end_time, success)
    
    def save_pipeline_summary(self, start_time: datetime, end_time: datetime, success: bool) -> None:
        """Save pipeline summary to JSON file."""
        try:
            summary = {
                'execution_id': start_time.strftime('%Y%m%d_%H%M%S'),
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': (end_time - start_time).total_seconds(),
                'success': success,
                'dry_run': self.dry_run,
                'stages': []
            }
            
            for stage in self.pipeline_stages:
                stage_data = {
                    'name': stage.name,
                    'description': stage.description,
                    'start_time': stage.start_time.isoformat() if stage.start_time else None,
                    'end_time': stage.end_time.isoformat() if stage.end_time else None,
                    'duration_seconds': stage.duration,
                    'success': stage.success,
                    'error_message': stage.error_message,
                    'result_data': stage.result_data
                }
                summary['stages'].append(stage_data)
            
            # Save to file
            summary_dir = Path("logs/pipeline_summaries")
            summary_dir.mkdir(parents=True, exist_ok=True)
            
            summary_file = summary_dir / f"pipeline_{summary['execution_id']}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
                
            logger.info(f"📄 Pipeline summary saved: {summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline summary: {e}")
    
    def send_error_notification(self) -> None:
        """Send email notification about pipeline failure."""
        if not self.email_config['email_user'] or not self.email_config['notification_emails']:
            logger.warning("Email configuration incomplete - skipping error notification")
            return
        
        try:
            # Get failed stages
            failed_stages = [stage for stage in self.pipeline_stages if not stage.success]
            
            if not failed_stages:
                return
            
            # Create email content
            subject = f"🚨 Autonomous Trading System: Pipeline Failed"
            
            body = f"""
            <html>
            <body>
                <h2>🚨 Pipeline Execution Failed</h2>
                <p><strong>Execution Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Mode:</strong> {'Dry Run' if self.dry_run else 'Live Trading'}</p>
                
                <h3>Failed Stages:</h3>
                <ul>
            """
            
            for stage in failed_stages:
                body += f"""
                    <li>
                        <strong>{stage.name}</strong>: {stage.description}<br>
                        <em>Error:</em> {stage.error_message}<br>
                        <em>Duration:</em> {stage.duration:.1f}s
                    </li>
                """
            
            body += """
                </ul>
                
                <h3>All Stages:</h3>
                <table border="1" style="border-collapse: collapse;">
                    <tr style="background-color: #f0f0f0;">
                        <th>Stage</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Error</th>
                    </tr>
            """
            
            for stage in self.pipeline_stages:
                status = "✅ Success" if stage.success else "❌ Failed"
                error = stage.error_message if stage.error_message else ""
                
                body += f"""
                    <tr>
                        <td>{stage.name}</td>
                        <td>{status}</td>
                        <td>{stage.duration:.1f}s</td>
                        <td>{error}</td>
                    </tr>
                """
            
            body += """
                </table>
                
                <p><strong>Please check the system logs and dashboard for more details.</strong></p>
                <p>Dashboard: <a href="http://127.0.0.1:8050">http://127.0.0.1:8050</a></p>
                
                <p>Autonomous Trading System</p>
            </body>
            </html>
            """
            
            # Send email
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email_user']
            msg['To'] = ', '.join(self.email_config['notification_emails'])
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))
            
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email_user'], self.email_config['email_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("📧 Error notification email sent")
            
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")


def run_scheduled_pipeline():
    """Function called by scheduler."""
    orchestrator = TradingSystemOrchestrator(dry_run=False)
    orchestrator.run_full_pipeline()


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Autonomous Trading System Orchestrator')
    parser.add_argument('--schedule', action='store_true', 
                       help='Run in scheduled mode (Sunday 6 PM)')
    parser.add_argument('--force', action='store_true',
                       help='Force immediate pipeline execution')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no actual trades)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set logging level')
    
    args = parser.parse_args()
    
    # Set up logging
    global logger
    logger = setup_logging(args.log_level)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logger.info("🤖 Autonomous Trading System Orchestrator Starting")
    logger.info(f"📅 Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.force:
        # Force immediate execution
        logger.info("🚀 Force execution requested")
        orchestrator = TradingSystemOrchestrator(dry_run=args.dry_run)
        success = orchestrator.run_full_pipeline()
        sys.exit(0 if success else 1)
    
    elif args.schedule:
        # Run in scheduled mode
        logger.info("⏰ Running in scheduled mode")
        logger.info("📅 Scheduled execution: Every Sunday at 6:00 PM")
        
        # Schedule the pipeline
        schedule.every().sunday.at("18:00").do(run_scheduled_pipeline)
        
        logger.info("⏳ Waiting for scheduled execution...")
        logger.info("🛑 Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            logger.info("🛑 Scheduler stopped by user")
            
    else:
        # Show help and current status
        print("\n🤖 Autonomous Trading System Orchestrator")
        print("=" * 50)
        print("Usage options:")
        print("  --schedule    Run scheduled mode (Sunday 6 PM)")
        print("  --force       Force immediate execution")
        print("  --dry-run     Run without executing trades")
        print("  --log-level   Set logging level (DEBUG/INFO/WARNING/ERROR)")
        print()
        print("Examples:")
        print("  python src/main.py --schedule")
        print("  python src/main.py --force --dry-run")
        print("  python src/main.py --force --log-level DEBUG")
        print()
        
        # Show next scheduled time if running
        next_run = schedule.next_run()
        if next_run:
            print(f"Next scheduled run: {next_run}")
        else:
            print("No scheduled runs configured")


if __name__ == "__main__":
    main()