#!/usr/bin/env python3
"""
Master Signal Aggregator
Combines multiple signal sources with configurable weights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_analyst import LLMAnalyst
from news_collector import NewsCollector
from data_collector import MarketDataCollector
from signals.technical_signals import TechnicalSignalGenerator

logger = logging.getLogger(__name__)


class SignalAggregator:
    """Master signal aggregator combining multiple signal sources."""
    
    def __init__(self, 
                 cache_dir: str = "data/signals",
                 default_weights: Optional[Dict[str, float]] = None):
        """Initialize the signal aggregator.
        
        Args:
            cache_dir: Directory to cache aggregated signals
            default_weights: Default weights for signal combination
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Default signal weights
        self.default_weights = default_weights or {
            'llm_sentiment': 0.30,
            'technical_momentum': 0.40,
            'volatility_regime': 0.20,
            'sector_rotation': 0.10
        }
        
        # Initialize component modules
        self.llm_analyst = LLMAnalyst()
        self.news_collector = NewsCollector()
        self.data_collector = MarketDataCollector()
        self.technical_generator = TechnicalSignalGenerator()
        
        # Market regime adjustments
        self.regime_adjustments = {
            'Risk-On': {
                'llm_sentiment': 1.2,    # Weight sentiment more in risk-on
                'technical_momentum': 1.1,
                'volatility_regime': 0.8,
                'sector_rotation': 1.0
            },
            'Risk-Off': {
                'llm_sentiment': 0.8,    # Weight sentiment less in risk-off
                'technical_momentum': 0.9,
                'volatility_regime': 1.3,  # Weight volatility more
                'sector_rotation': 1.1
            },
            'Neutral': {
                'llm_sentiment': 1.0,
                'technical_momentum': 1.0,
                'volatility_regime': 1.0,
                'sector_rotation': 1.0
            }
        }
    
    def normalize_signal(self, value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a signal to 0-1 scale.
        
        Args:
            value: Value to normalize
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value between 0 and 1
        """
        if max_val <= min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        return max(0.0, min(1.0, normalized))
    
    def extract_llm_sentiment_signal(self, llm_analysis: Dict[str, Any]) -> float:
        """Extract normalized sentiment signal from LLM analysis.
        
        Args:
            llm_analysis: LLM analysis results
            
        Returns:
            Normalized sentiment signal (0-1)
        """
        if not llm_analysis or 'sentiment_score' not in llm_analysis:
            return 0.5
        
        sentiment = llm_analysis['sentiment_score']  # Already -1 to +1
        confidence = llm_analysis.get('confidence', 0.5)
        
        # Convert sentiment to 0-1 scale
        normalized_sentiment = (sentiment + 1) / 2
        
        # Weight by confidence
        weighted_signal = normalized_sentiment * confidence + 0.5 * (1 - confidence)
        
        return max(0.0, min(1.0, weighted_signal))
    
    def extract_technical_momentum_signal(self, technical_signals: Dict[str, float]) -> float:
        """Extract normalized momentum signal from technical analysis.
        
        Args:
            technical_signals: Technical signal results
            
        Returns:
            Normalized momentum signal (0-1)
        """
        if not technical_signals:
            return 0.5
        
        # Primary momentum signal
        momentum_strength = technical_signals.get('momentum_strength', 0.5)
        
        # Adjust based on RSI positioning
        rsi_signal = technical_signals.get('rsi_opportunity', 0.5)
        
        # Combine momentum with RSI opportunity
        combined_signal = momentum_strength * 0.8 + rsi_signal * 0.2
        
        return max(0.0, min(1.0, combined_signal))
    
    def extract_volatility_regime_signal(self, technical_signals: Dict[str, float]) -> float:
        """Extract normalized volatility regime signal.
        
        Args:
            technical_signals: Technical signal results
            
        Returns:
            Normalized volatility signal (0-1)
        """
        if not technical_signals:
            return 0.5
        
        volatility_regime = technical_signals.get('volatility_regime', 0.5)
        atr_signal = technical_signals.get('atr_signal', 0.5)
        volatility_opportunity = technical_signals.get('volatility_opportunity', 0.5)
        
        # Combine volatility components
        combined_signal = (
            volatility_regime * 0.5 +
            atr_signal * 0.3 +
            volatility_opportunity * 0.2
        )
        
        return max(0.0, min(1.0, combined_signal))
    
    def extract_sector_rotation_signal(self, technical_signals: Dict[str, float]) -> float:
        """Extract normalized sector rotation signal.
        
        Args:
            technical_signals: Technical signal results
            
        Returns:
            Normalized sector rotation signal (0-1)
        """
        if not technical_signals:
            return 0.5
        
        sector_relative = technical_signals.get('sector_relative', 0.5)
        sector_momentum = technical_signals.get('sector_momentum', 0.5)
        
        # Combine sector signals
        combined_signal = sector_relative * 0.7 + sector_momentum * 0.3
        
        return max(0.0, min(1.0, combined_signal))
    
    def apply_regime_adjustments(self, 
                                weights: Dict[str, float], 
                                market_regime: str) -> Dict[str, float]:
        """Apply market regime adjustments to signal weights.
        
        Args:
            weights: Original signal weights
            market_regime: Current market regime
            
        Returns:
            Adjusted weights
        """
        adjustments = self.regime_adjustments.get(market_regime, self.regime_adjustments['Neutral'])
        
        adjusted_weights = {}
        for signal, weight in weights.items():
            adjustment = adjustments.get(signal, 1.0)
            adjusted_weights[signal] = weight * adjustment
        
        # Renormalize to sum to 1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def generate_aggregated_signals(self, 
                                  tickers: List[str],
                                  weights: Optional[Dict[str, float]] = None,
                                  use_regime_adjustment: bool = True) -> pd.DataFrame:
        """Generate aggregated signals for multiple stocks.
        
        Args:
            tickers: List of stock tickers
            weights: Custom signal weights
            use_regime_adjustment: Whether to apply market regime adjustments
            
        Returns:
            DataFrame with aggregated signals and rankings
        """
        if weights is None:
            weights = self.default_weights.copy()
        
        logger.info(f"Generating aggregated signals for {len(tickers)} stocks")
        
        # 1. Collect market data
        stock_data = {}
        for ticker in tickers:
            df = self.data_collector.load_from_parquet(ticker)
            if df is not None and not df.empty:
                stock_data[ticker] = df
        
        if not stock_data:
            logger.error("No market data available")
            return pd.DataFrame()
        
        # 2. Generate technical signals
        logger.info("Generating technical signals...")
        technical_signals_df = self.technical_generator.generate_all_signals(stock_data)
        
        # 3. Get market regime
        market_regime = 'Neutral'
        regime_confidence = 0.5
        
        if use_regime_adjustment:
            try:
                market_summary = self.llm_analyst.get_market_summary(stock_data)
                regime_analysis = self.llm_analyst.analyze_market_regime(market_summary)
                market_regime = regime_analysis.get('regime', 'Neutral')
                regime_confidence = regime_analysis.get('confidence', 0.5)
                logger.info(f"Market regime: {market_regime} (confidence: {regime_confidence:.2f})")
            except Exception as e:
                logger.warning(f"Could not determine market regime: {e}")
        
        # 4. Apply regime adjustments
        if use_regime_adjustment:
            adjusted_weights = self.apply_regime_adjustments(weights, market_regime)
            logger.info(f"Adjusted weights for {market_regime} regime: {adjusted_weights}")
        else:
            adjusted_weights = weights
        
        # 5. Collect news and generate LLM sentiment signals
        logger.info("Collecting news and generating LLM sentiment...")
        news_data = {}
        llm_signals = {}
        
        # Use ThreadPoolExecutor for parallel LLM analysis
        def analyze_stock_sentiment(ticker):
            try:
                # Get recent news
                news = self.news_collector.get_weekly_news(ticker, weeks_back=1)
                if not news:
                    news = self.news_collector.collect_stock_news(ticker, use_cache=True)
                
                if news:
                    # Analyze sentiment
                    sentiment_analysis = self.llm_analyst.analyze_news_sentiment(ticker, news[:10])
                    return ticker, sentiment_analysis
                else:
                    return ticker, None
            except Exception as e:
                logger.error(f"Error analyzing sentiment for {ticker}: {e}")
                return ticker, None
        
        # Process sentiment analysis in parallel (limited workers to avoid overwhelming LLM)
        sentiment_results = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_ticker = {executor.submit(analyze_stock_sentiment, ticker): ticker 
                              for ticker in tickers if ticker in stock_data}
            
            for future in as_completed(future_to_ticker):
                ticker, result = future.result()
                sentiment_results[ticker] = result
        
        # 6. Aggregate all signals
        results = []
        
        for _, tech_row in technical_signals_df.iterrows():
            ticker = tech_row['ticker']
            
            # Extract individual signals
            llm_sentiment_signal = 0.5
            if ticker in sentiment_results and sentiment_results[ticker]:
                llm_sentiment_signal = self.extract_llm_sentiment_signal(sentiment_results[ticker])
            
            technical_momentum_signal = self.extract_technical_momentum_signal(tech_row.to_dict())
            volatility_regime_signal = self.extract_volatility_regime_signal(tech_row.to_dict())
            sector_rotation_signal = self.extract_sector_rotation_signal(tech_row.to_dict())
            
            # Calculate weighted aggregate score
            aggregate_score = (
                llm_sentiment_signal * adjusted_weights['llm_sentiment'] +
                technical_momentum_signal * adjusted_weights['technical_momentum'] +
                volatility_regime_signal * adjusted_weights['volatility_regime'] +
                sector_rotation_signal * adjusted_weights['sector_rotation']
            )
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                ticker,
                llm_sentiment_signal,
                technical_momentum_signal,
                volatility_regime_signal,
                sector_rotation_signal,
                adjusted_weights,
                sentiment_results.get(ticker),
                tech_row.to_dict()
            )
            
            result = {
                'ticker': ticker,
                'aggregate_score': round(aggregate_score, 4),
                'llm_sentiment_signal': round(llm_sentiment_signal, 4),
                'technical_momentum_signal': round(technical_momentum_signal, 4),
                'volatility_regime_signal': round(volatility_regime_signal, 4),
                'sector_rotation_signal': round(sector_rotation_signal, 4),
                'market_regime': market_regime,
                'regime_confidence': regime_confidence,
                'current_price': tech_row.get('current_price', 0),
                'rsi': tech_row.get('rsi', 50),
                'reasoning': reasoning,
                'timestamp': datetime.now().isoformat()
            }
            
            results.append(result)
        
        # Create DataFrame and rank stocks
        signals_df = pd.DataFrame(results)
        
        if not signals_df.empty:
            signals_df = signals_df.sort_values('aggregate_score', ascending=False)
            signals_df['rank'] = range(1, len(signals_df) + 1)
            
            # Add percentile rankings
            signals_df['percentile'] = signals_df['aggregate_score'].rank(pct=True)
        
        # Cache results
        self._cache_signals(signals_df, weights, market_regime)
        
        return signals_df
    
    def _generate_reasoning(self,
                          ticker: str,
                          llm_sentiment: float,
                          technical_momentum: float,
                          volatility_regime: float,
                          sector_rotation: float,
                          weights: Dict[str, float],
                          sentiment_analysis: Optional[Dict],
                          technical_data: Dict) -> str:
        """Generate detailed reasoning for the aggregate score.
        
        Args:
            ticker: Stock ticker
            llm_sentiment: LLM sentiment signal
            technical_momentum: Technical momentum signal
            volatility_regime: Volatility regime signal
            sector_rotation: Sector rotation signal
            weights: Signal weights used
            sentiment_analysis: Full sentiment analysis
            technical_data: Technical indicators data
            
        Returns:
            Detailed reasoning string
        """
        reasons = []
        
        # LLM Sentiment reasoning
        if llm_sentiment > 0.6:
            if sentiment_analysis and sentiment_analysis.get('positive_factors'):
                reasons.append(f"Positive news sentiment ({llm_sentiment:.2f}): {', '.join(sentiment_analysis['positive_factors'][:2])}")
            else:
                reasons.append(f"Strong positive sentiment ({llm_sentiment:.2f})")
        elif llm_sentiment < 0.4:
            if sentiment_analysis and sentiment_analysis.get('negative_factors'):
                reasons.append(f"Negative news sentiment ({llm_sentiment:.2f}): {', '.join(sentiment_analysis['negative_factors'][:2])}")
            else:
                reasons.append(f"Weak sentiment ({llm_sentiment:.2f})")
        
        # Technical momentum reasoning
        if technical_momentum > 0.6:
            price_sma20 = technical_data.get('price_to_sma20', 1.0)
            if price_sma20 > 1.05:
                reasons.append(f"Strong technical momentum ({technical_momentum:.2f}): Price {((price_sma20-1)*100):+.1f}% vs SMA20")
            else:
                reasons.append(f"Strong technical momentum ({technical_momentum:.2f})")
        elif technical_momentum < 0.4:
            reasons.append(f"Weak technical momentum ({technical_momentum:.2f})")
        
        # Volatility reasoning
        if volatility_regime > 0.6:
            reasons.append(f"Favorable volatility environment ({volatility_regime:.2f})")
        elif volatility_regime < 0.4:
            atr_pct = technical_data.get('atr_percent', 2.0)
            reasons.append(f"High volatility concern ({volatility_regime:.2f}): ATR {atr_pct:.1f}%")
        
        # Sector rotation reasoning
        if sector_rotation > 0.6:
            reasons.append(f"Outperforming sector ({sector_rotation:.2f})")
        elif sector_rotation < 0.4:
            reasons.append(f"Underperforming vs sector ({sector_rotation:.2f})")
        
        # Weight impact
        dominant_signal = max(weights.items(), key=lambda x: x[1])
        reasons.append(f"Weighted by {dominant_signal[0].replace('_', ' ')} ({dominant_signal[1]:.0%})")
        
        return " | ".join(reasons) if reasons else "No significant signals"
    
    def select_top_candidates(self, 
                            signals_df: pd.DataFrame,
                            top_n: int = 8,
                            min_score: float = 0.5) -> pd.DataFrame:
        """Select top stock candidates based on aggregate scores.
        
        Args:
            signals_df: DataFrame with aggregated signals
            top_n: Number of top stocks to select
            min_score: Minimum aggregate score threshold
            
        Returns:
            DataFrame with top candidates
        """
        if signals_df.empty:
            return pd.DataFrame()
        
        # Filter by minimum score
        filtered = signals_df[signals_df['aggregate_score'] >= min_score].copy()
        
        # Select top N
        top_candidates = filtered.head(top_n)
        
        logger.info(f"Selected {len(top_candidates)} candidates (min_score: {min_score})")
        
        return top_candidates
    
    def backtest_weight_combinations(self, 
                                   tickers: List[str],
                                   lookback_days: int = 60,
                                   top_n: int = 5) -> Dict[str, Any]:
        """Backtest different weight combinations.
        
        Args:
            tickers: List of tickers to test
            lookback_days: Days to backtest
            top_n: Number of top stocks to select for each test
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Backtesting weight combinations over {lookback_days} days")
        
        # Define weight combinations to test
        weight_combinations = [
            # Default
            {'llm_sentiment': 0.30, 'technical_momentum': 0.40, 'volatility_regime': 0.20, 'sector_rotation': 0.10},
            # Technical-heavy
            {'llm_sentiment': 0.20, 'technical_momentum': 0.50, 'volatility_regime': 0.20, 'sector_rotation': 0.10},
            # Sentiment-heavy
            {'llm_sentiment': 0.50, 'technical_momentum': 0.30, 'volatility_regime': 0.10, 'sector_rotation': 0.10},
            # Balanced
            {'llm_sentiment': 0.25, 'technical_momentum': 0.25, 'volatility_regime': 0.25, 'sector_rotation': 0.25},
            # Volatility-focused
            {'llm_sentiment': 0.25, 'technical_momentum': 0.35, 'volatility_regime': 0.30, 'sector_rotation': 0.10},
        ]
        
        results = {}
        
        for i, weights in enumerate(weight_combinations):
            weight_name = f"Combination_{i+1}"
            logger.info(f"Testing {weight_name}: {weights}")
            
            try:
                # Generate signals with these weights (simplified backtest)
                signals_df = self.generate_aggregated_signals(
                    tickers, 
                    weights=weights, 
                    use_regime_adjustment=False  # Simplify for backtest
                )
                
                if signals_df.empty:
                    continue
                
                # Select top candidates
                top_candidates = self.select_top_candidates(signals_df, top_n=top_n)
                
                # Simple performance metric (average of top scores)
                avg_score = top_candidates['aggregate_score'].mean() if not top_candidates.empty else 0
                score_std = top_candidates['aggregate_score'].std() if len(top_candidates) > 1 else 0
                
                results[weight_name] = {
                    'weights': weights,
                    'avg_score': avg_score,
                    'score_std': score_std,
                    'top_candidates': len(top_candidates),
                    'consistency': avg_score / (score_std + 0.001)  # Higher is better
                }
                
            except Exception as e:
                logger.error(f"Error testing {weight_name}: {e}")
                results[weight_name] = {'error': str(e)}
        
        # Rank combinations by consistency
        valid_results = {k: v for k, v in results.items() if 'error' not in v}
        if valid_results:
            best_combination = max(valid_results.items(), key=lambda x: x[1]['consistency'])
            results['best_combination'] = best_combination[0]
            results['best_weights'] = best_combination[1]['weights']
        
        return results
    
    def _cache_signals(self, signals_df: pd.DataFrame, weights: Dict[str, float], market_regime: str) -> None:
        """Cache aggregated signals.
        
        Args:
            signals_df: Signals DataFrame
            weights: Weights used
            market_regime: Market regime
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cache_file = self.cache_dir / f"signals_{timestamp}.json"
            
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'weights': weights,
                'market_regime': market_regime,
                'signals': signals_df.to_dict('records')
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
            logger.info(f"Cached signals to {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching signals: {e}")


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Signal Aggregator')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    parser.add_argument('--top', type=int, default=8, help='Number of top candidates to select')
    parser.add_argument('--backtest', action='store_true', help='Backtest weight combinations')
    parser.add_argument('--weights', help='Custom weights as JSON string')
    
    args = parser.parse_args()
    
    # Initialize aggregator
    aggregator = SignalAggregator()
    
    # Get tickers
    if args.tickers:
        tickers = args.tickers
    else:
        tickers = MarketDataCollector.TECH_UNIVERSE[:15]  # Top 15 for demo
    
    # Custom weights
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError:
            print("Invalid JSON for weights")
            return
    
    if args.backtest:
        # Backtest weight combinations
        print("Backtesting weight combinations...")
        results = aggregator.backtest_weight_combinations(tickers)
        
        print("\nBacktest Results:")
        print("-" * 80)
        for combo_name, result in results.items():
            if combo_name.startswith('Combination'):
                print(f"\n{combo_name}:")
                print(f"  Avg Score: {result['avg_score']:.4f}")
                print(f"  Consistency: {result['consistency']:.3f}")
                print(f"  Weights: {result['weights']}")
        
        if 'best_combination' in results:
            print(f"\nBest Combination: {results['best_combination']}")
            print(f"Best Weights: {results['best_weights']}")
    
    else:
        # Generate aggregated signals
        print(f"Generating aggregated signals for {len(tickers)} stocks...")
        signals_df = aggregator.generate_aggregated_signals(tickers, weights=weights)
        
        if signals_df.empty:
            print("No signals generated. Check data availability.")
            return
        
        # Select top candidates
        top_candidates = aggregator.select_top_candidates(signals_df, top_n=args.top)
        
        # Display results
        print("\n" + "="*100)
        print("AGGREGATED SIGNAL RESULTS")
        print("="*100)
        
        print(f"\nMarket Regime: {signals_df.iloc[0]['market_regime']} "
              f"(confidence: {signals_df.iloc[0]['regime_confidence']:.2f})")
        
        print(f"\nTop {len(top_candidates)} Candidates:")
        print("-" * 100)
        
        for _, row in top_candidates.iterrows():
            print(f"\n{row['rank']}. {row['ticker']} - Score: {row['aggregate_score']:.3f}")
            print(f"   LLM: {row['llm_sentiment_signal']:.3f} | "
                  f"Tech: {row['technical_momentum_signal']:.3f} | "
                  f"Vol: {row['volatility_regime_signal']:.3f} | "
                  f"Sector: {row['sector_rotation_signal']:.3f}")
            print(f"   Reasoning: {row['reasoning']}")
            print(f"   Price: ${row['current_price']:.2f} | RSI: {row['rsi']:.1f}")


if __name__ == "__main__":
    main()