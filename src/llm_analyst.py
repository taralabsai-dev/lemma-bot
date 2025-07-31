#!/usr/bin/env python3
"""
LLM Analysis Module
Uses Ollama to analyze market data and news for trading insights
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMAnalyst:
    """Analyzes market data and news using Ollama LLM."""
    
    def __init__(self, 
                 model_name: str = "llama3.2:3b",
                 api_url: str = "http://localhost:11434",
                 cache_dir: str = "data/analysis"):
        """Initialize LLM analyst.
        
        Args:
            model_name: Ollama model to use
            api_url: Ollama API URL
            cache_dir: Directory to cache analysis results
        """
        self.model_name = model_name
        self.api_url = api_url
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Analysis parameters
        self.temperature = 0.3  # Lower for more consistent analysis
        self.max_retries = 3
        self.retry_delay = 2.0
        self.timeout = 30
        
        # Market regime cache
        self.regime_cache_dir = self.cache_dir / "regimes"
        self.regime_cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_cache_key(self, analysis_type: str, ticker: str, data_hash: str) -> str:
        """Generate cache key for analysis results."""
        content = f"{analysis_type}_{ticker}_{data_hash}_{self.model_name}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load analysis from cache.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached analysis or None
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                    # Check if cache is still valid (24 hours)
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - cached_time < timedelta(hours=24):
                        return data
            except Exception as e:
                logger.error(f"Error loading cache {cache_key}: {e}")
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Save analysis to cache.
        
        Args:
            cache_key: Cache key
            data: Analysis data to save
        """
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            data['timestamp'] = datetime.now().isoformat()
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache {cache_key}: {e}")
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call Ollama API with retry logic.
        
        Args:
            prompt: Prompt to send to Ollama
            
        Returns:
            Response text or None on error
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": self.temperature,
                            "top_p": 0.9,
                            "max_tokens": 500
                        }
                    },
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')
                else:
                    logger.error(f"Ollama API error: {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Ollama timeout (attempt {attempt + 1}/{self.max_retries})")
            except Exception as e:
                logger.error(f"Ollama error: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay * (attempt + 1))
        
        return None
    
    def analyze_news_sentiment(self, ticker: str, headlines: List[Dict]) -> Dict[str, Any]:
        """Analyze news sentiment for a stock.
        
        Args:
            ticker: Stock ticker symbol
            headlines: List of news headlines with metadata
            
        Returns:
            Analysis results with sentiment score and insights
        """
        # Create data hash for caching
        headlines_text = "\n".join([h['headline'] for h in headlines])
        data_hash = hashlib.md5(headlines_text.encode()).hexdigest()[:16]
        
        # Check cache
        cache_key = self._generate_cache_key("news_sentiment", ticker, data_hash)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info(f"Using cached news sentiment analysis for {ticker}")
            return cached_result
        
        # Prepare headlines for analysis
        headlines_formatted = []
        for i, item in enumerate(headlines[:15], 1):  # Limit to 15 most recent
            date = item['timestamp'][:10]
            source = item['source']
            headline = item['headline']
            headlines_formatted.append(f"{i}. [{date}] {source}: {headline}")
        
        headlines_text = "\n".join(headlines_formatted)
        
        # Create prompt
        prompt = f"""You are a financial analyst specializing in technology stocks. Analyze these recent news headlines for {ticker} and provide a structured analysis.

Headlines:
{headlines_text}

Provide your analysis in the following JSON format:
{{
    "sentiment_score": <number between -1.0 and 1.0>,
    "confidence": <number between 0.0 and 1.0>,
    "key_themes": [<list of 3-5 main themes>],
    "positive_factors": [<list of positive points>],
    "negative_factors": [<list of negative points>],
    "trading_implications": "<brief trading recommendation>",
    "risk_factors": [<list of key risks>]
}}

Focus on:
1. Overall sentiment (-1 very negative, 0 neutral, +1 very positive)
2. Key themes and trends in the news
3. Potential impact on stock price
4. Trading implications for the next 1-2 weeks

Respond ONLY with the JSON object, no additional text."""

        # Get LLM response
        logger.info(f"Analyzing news sentiment for {ticker}")
        response = self._call_ollama(prompt)
        
        if response:
            try:
                # Parse JSON response
                analysis = json.loads(response)
                
                # Validate and clean the response
                result = {
                    'ticker': ticker,
                    'sentiment_score': float(analysis.get('sentiment_score', 0)),
                    'confidence': float(analysis.get('confidence', 0.5)),
                    'key_themes': analysis.get('key_themes', [])[:5],
                    'positive_factors': analysis.get('positive_factors', [])[:5],
                    'negative_factors': analysis.get('negative_factors', [])[:5],
                    'trading_implications': analysis.get('trading_implications', 'No clear signal'),
                    'risk_factors': analysis.get('risk_factors', [])[:3],
                    'headlines_analyzed': len(headlines_formatted),
                    'analysis_date': datetime.now().isoformat()
                }
                
                # Save to cache
                self._save_to_cache(cache_key, result)
                
                return result
                
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response for {ticker}")
                # Try to extract sentiment from text
                sentiment = 0.0
                if "positive" in response.lower():
                    sentiment += 0.3
                if "negative" in response.lower():
                    sentiment -= 0.3
                if "bullish" in response.lower():
                    sentiment += 0.2
                if "bearish" in response.lower():
                    sentiment -= 0.2
                
                return {
                    'ticker': ticker,
                    'sentiment_score': max(-1, min(1, sentiment)),
                    'confidence': 0.3,
                    'key_themes': ['Analysis parsing error'],
                    'positive_factors': [],
                    'negative_factors': [],
                    'trading_implications': 'Unable to parse full analysis',
                    'risk_factors': ['LLM response parsing failed'],
                    'headlines_analyzed': len(headlines_formatted),
                    'analysis_date': datetime.now().isoformat()
                }
        
        # Fallback if LLM fails
        return {
            'ticker': ticker,
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'key_themes': ['LLM analysis failed'],
            'positive_factors': [],
            'negative_factors': [],
            'trading_implications': 'No analysis available',
            'risk_factors': ['Analysis unavailable'],
            'headlines_analyzed': 0,
            'analysis_date': datetime.now().isoformat()
        }
    
    def analyze_technical_indicators(self, ticker: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze technical indicators for trading signals.
        
        Args:
            ticker: Stock ticker symbol
            market_data: DataFrame with OHLCV and technical indicators
            
        Returns:
            Technical analysis results
        """
        # Get latest data
        latest = market_data.iloc[-1]
        recent = market_data.tail(5)
        
        # Create data hash for caching
        data_str = f"{ticker}_{latest['Close']}_{latest['RSI_14']}_{latest['Volume']}"
        data_hash = hashlib.md5(data_str.encode()).hexdigest()[:16]
        
        # Check cache
        cache_key = self._generate_cache_key("technical", ticker, data_hash)
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info(f"Using cached technical analysis for {ticker}")
            return cached_result
        
        # Prepare technical data summary
        technical_summary = f"""
Current Price: ${latest['Close']:.2f}
SMA 20: ${latest['SMA_20']:.2f} ({((latest['Close']/latest['SMA_20']-1)*100):.1f}% vs price)
SMA 50: ${latest['SMA_50']:.2f} ({((latest['Close']/latest['SMA_50']-1)*100):.1f}% vs price)
RSI (14): {latest['RSI_14']:.1f}
ATR %: {latest['ATR_Percent']:.2f}%
Volume Ratio: {latest['Volume_Ratio']:.2f}x average
5-day price change: {((latest['Close']/recent.iloc[0]['Close']-1)*100):.1f}%
20-day volatility: {latest['Volatility_20']:.1f}%
"""

        # Create prompt
        prompt = f"""You are a technical analyst specializing in technology stocks. Analyze the technical indicators for {ticker} and provide a structured analysis.

{technical_summary}

Market conditions:
- RSI > 70 indicates overbought, < 30 indicates oversold
- Price above both SMAs suggests uptrend
- High ATR% indicates high volatility
- Volume ratio > 1.5 suggests increased interest

Provide your analysis in the following JSON format:
{{
    "technical_signal": "<BUY/HOLD/SELL>",
    "signal_strength": <number between 0.0 and 1.0>,
    "trend": "<BULLISH/NEUTRAL/BEARISH>",
    "support_level": <estimated support price>,
    "resistance_level": <estimated resistance price>,
    "key_observations": [<list of 3-5 key technical observations>],
    "entry_points": [<list of potential entry prices/conditions>],
    "exit_points": [<list of potential exit prices/conditions>]
}}

Focus on actionable insights for short-term trading (1-2 weeks).
Respond ONLY with the JSON object."""

        # Get LLM response
        logger.info(f"Analyzing technical indicators for {ticker}")
        response = self._call_ollama(prompt)
        
        if response:
            try:
                # Parse JSON response
                analysis = json.loads(response)
                
                # Validate and clean the response
                result = {
                    'ticker': ticker,
                    'current_price': float(latest['Close']),
                    'technical_signal': analysis.get('technical_signal', 'HOLD'),
                    'signal_strength': float(analysis.get('signal_strength', 0.5)),
                    'trend': analysis.get('trend', 'NEUTRAL'),
                    'support_level': float(analysis.get('support_level', latest['Close'] * 0.95)),
                    'resistance_level': float(analysis.get('resistance_level', latest['Close'] * 1.05)),
                    'key_observations': analysis.get('key_observations', [])[:5],
                    'entry_points': analysis.get('entry_points', [])[:3],
                    'exit_points': analysis.get('exit_points', [])[:3],
                    'rsi': float(latest['RSI_14']),
                    'volume_ratio': float(latest['Volume_Ratio']),
                    'analysis_date': datetime.now().isoformat()
                }
                
                # Save to cache
                self._save_to_cache(cache_key, result)
                
                return result
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Failed to parse technical analysis for {ticker}: {e}")
        
        # Fallback analysis based on simple rules
        signal = 'HOLD'
        if latest['RSI_14'] < 30 and latest['Close'] > latest['SMA_20']:
            signal = 'BUY'
        elif latest['RSI_14'] > 70 or latest['Close'] < latest['SMA_50']:
            signal = 'SELL'
        
        return {
            'ticker': ticker,
            'current_price': float(latest['Close']),
            'technical_signal': signal,
            'signal_strength': 0.3,
            'trend': 'BULLISH' if latest['Close'] > latest['SMA_50'] else 'BEARISH',
            'support_level': float(latest['Close'] * 0.95),
            'resistance_level': float(latest['Close'] * 1.05),
            'key_observations': ['Fallback analysis - LLM unavailable'],
            'entry_points': [],
            'exit_points': [],
            'rsi': float(latest['RSI_14']),
            'volume_ratio': float(latest['Volume_Ratio']),
            'analysis_date': datetime.now().isoformat()
        }
    
    def combine_analysis(self, 
                        ticker: str,
                        news_analysis: Dict[str, Any],
                        technical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Combine news and technical analysis for final recommendation.
        
        Args:
            ticker: Stock ticker symbol
            news_analysis: News sentiment analysis results
            technical_analysis: Technical indicator analysis results
            
        Returns:
            Combined analysis with final recommendation
        """
        # Calculate combined score
        news_weight = 0.4
        technical_weight = 0.6
        
        # Normalize sentiment to 0-1 scale
        news_score = (news_analysis['sentiment_score'] + 1) / 2
        
        # Convert technical signal to score
        tech_score_map = {'BUY': 0.8, 'HOLD': 0.5, 'SELL': 0.2}
        tech_score = tech_score_map.get(technical_analysis['technical_signal'], 0.5)
        tech_score *= technical_analysis['signal_strength']
        
        # Combined score
        combined_score = (news_score * news_weight) + (tech_score * technical_weight)
        
        # Determine action
        if combined_score >= 0.65:
            action = 'BUY'
        elif combined_score <= 0.35:
            action = 'SELL'
        else:
            action = 'HOLD'
        
        # Confidence based on agreement
        if news_analysis['sentiment_score'] > 0.3 and technical_analysis['technical_signal'] == 'BUY':
            confidence = 0.8
        elif news_analysis['sentiment_score'] < -0.3 and technical_analysis['technical_signal'] == 'SELL':
            confidence = 0.8
        else:
            confidence = 0.5
        
        return {
            'ticker': ticker,
            'action': action,
            'combined_score': round(combined_score, 3),
            'confidence': confidence,
            'news_sentiment': news_analysis['sentiment_score'],
            'technical_signal': technical_analysis['technical_signal'],
            'current_price': technical_analysis['current_price'],
            'support_level': technical_analysis['support_level'],
            'resistance_level': technical_analysis['resistance_level'],
            'key_factors': {
                'positive': news_analysis['positive_factors'][:3] + technical_analysis['key_observations'][:2],
                'negative': news_analysis['negative_factors'][:3] + news_analysis['risk_factors'][:2]
            },
            'recommendation': f"{action} - Combined score: {combined_score:.2f}",
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def batch_analyze_stocks(self, 
                           stock_data: Dict[str, Tuple[List[Dict], pd.DataFrame]],
                           max_workers: int = 3) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple stocks in parallel.
        
        Args:
            stock_data: Dictionary mapping ticker to (news_list, market_df)
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary mapping ticker to combined analysis
        """
        results = {}
        
        def analyze_single_stock(ticker: str, news: List[Dict], market_df: pd.DataFrame) -> Tuple[str, Dict]:
            try:
                # Analyze news sentiment
                news_analysis = self.analyze_news_sentiment(ticker, news)
                
                # Analyze technical indicators
                technical_analysis = self.analyze_technical_indicators(ticker, market_df)
                
                # Combine analyses
                combined = self.combine_analysis(ticker, news_analysis, technical_analysis)
                
                return ticker, combined
                
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                return ticker, {
                    'ticker': ticker,
                    'action': 'HOLD',
                    'error': str(e),
                    'analysis_timestamp': datetime.now().isoformat()
                }
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for ticker, (news, market_df) in stock_data.items():
                future = executor.submit(analyze_single_stock, ticker, news, market_df)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                ticker, analysis = future.result()
                results[ticker] = analysis
                logger.info(f"Completed analysis for {ticker}: {analysis['action']}")
        
        return results
    
    def get_market_summary(self, stock_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate weekly market data summary for regime analysis.
        
        Args:
            stock_data: Dictionary mapping ticker to DataFrame
            
        Returns:
            Market summary statistics
        """
        summary = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'stocks_analyzed': len(stock_data),
            'performance': {},
            'volatility': {},
            'volume': {},
            'sector_performance': {}
        }
        
        # Define sector mappings
        sector_map = {
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
            'PYPL': 'Fintech'
        }
        
        weekly_returns = []
        weekly_volatilities = []
        volume_ratios = []
        sector_returns = {}
        
        for ticker, df in stock_data.items():
            if df is None or len(df) < 7:
                continue
                
            # Get last 5 trading days
            recent = df.tail(5)
            
            # Weekly performance
            weekly_return = (recent['Close'].iloc[-1] / recent['Close'].iloc[0] - 1) * 100
            weekly_returns.append(weekly_return)
            summary['performance'][ticker] = round(weekly_return, 2)
            
            # Volatility (using ATR percentage)
            volatility = recent['ATR_Percent'].mean()
            weekly_volatilities.append(volatility)
            summary['volatility'][ticker] = round(volatility, 2)
            
            # Volume analysis
            vol_ratio = recent['Volume_Ratio'].mean()
            volume_ratios.append(vol_ratio)
            summary['volume'][ticker] = round(vol_ratio, 2)
            
            # Sector performance
            sector = sector_map.get(ticker, 'Other')
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(weekly_return)
        
        # Aggregate statistics
        if weekly_returns:
            summary['market_performance'] = {
                'avg_return': round(sum(weekly_returns) / len(weekly_returns), 2),
                'median_return': round(sorted(weekly_returns)[len(weekly_returns)//2], 2),
                'positive_stocks': sum(1 for r in weekly_returns if r > 0),
                'negative_stocks': sum(1 for r in weekly_returns if r < 0),
                'best_performer': max(weekly_returns),
                'worst_performer': min(weekly_returns)
            }
            
            summary['market_volatility'] = {
                'avg_volatility': round(sum(weekly_volatilities) / len(weekly_volatilities), 2),
                'high_vol_stocks': sum(1 for v in weekly_volatilities if v > 3.0),
                'low_vol_stocks': sum(1 for v in weekly_volatilities if v < 1.5)
            }
            
            summary['market_volume'] = {
                'avg_volume_ratio': round(sum(volume_ratios) / len(volume_ratios), 2),
                'high_volume_stocks': sum(1 for v in volume_ratios if v > 1.5),
                'low_volume_stocks': sum(1 for v in volume_ratios if v < 0.8)
            }
        
        # Sector performance
        for sector, returns in sector_returns.items():
            if returns:
                summary['sector_performance'][sector] = {
                    'avg_return': round(sum(returns) / len(returns), 2),
                    'stock_count': len(returns)
                }
        
        return summary
    
    def analyze_market_regime(self, market_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market regime using LLM.
        
        Args:
            market_summary: Weekly market data summary
            
        Returns:
            Market regime analysis
        """
        # Check cache
        date_str = market_summary['date']
        cache_key = f"regime_{date_str}_{self.model_name}"
        cached_result = self._load_from_cache(cache_key)
        if cached_result:
            logger.info(f"Using cached market regime analysis for {date_str}")
            return cached_result
        
        # Prepare market data for analysis
        perf = market_summary['market_performance']
        vol = market_summary['market_volatility']
        volume = market_summary['market_volume']
        sectors = market_summary['sector_performance']
        
        # Format sector performance
        sector_text = []
        for sector, data in sectors.items():
            sector_text.append(f"- {sector}: {data['avg_return']:+.1f}% ({data['stock_count']} stocks)")
        
        market_data_text = f"""
MARKET PERFORMANCE SUMMARY (Week ending {date_str}):

Overall Market:
- Average Return: {perf['avg_return']:+.1f}%
- Median Return: {perf['median_return']:+.1f}%
- Stocks Up: {perf['positive_stocks']} | Stocks Down: {perf['negative_stocks']}
- Best Performer: {perf['best_performer']:+.1f}%
- Worst Performer: {perf['worst_performer']:+.1f}%

Volatility Profile:
- Average Volatility: {vol['avg_volatility']:.1f}%
- High Volatility Stocks: {vol['high_vol_stocks']}
- Low Volatility Stocks: {vol['low_vol_stocks']}

Volume Activity:
- Average Volume Ratio: {volume['avg_volume_ratio']:.2f}x
- High Volume Stocks: {volume['high_volume_stocks']}
- Low Volume Stocks: {volume['low_volume_stocks']}

Sector Performance:
{chr(10).join(sector_text)}
"""

        # Create structured prompt for regime classification
        prompt = f"""You are a senior market strategist analyzing the current market regime for technology stocks. Based on the market data below, provide a structured analysis.

{market_data_text}

MARKET REGIME CLASSIFICATION CRITERIA:

**Risk-On Regime:**
- Broad-based gains (>60% stocks positive)
- Average returns > +1.0%
- High volume activity (>1.3x average)
- Growth sectors (Software, Semiconductors) outperforming
- Lower volatility environment

**Risk-Off Regime:**
- Broad-based declines (>60% stocks negative)
- Average returns < -1.0%
- Defensive rotation or cash hoarding
- High volatility (>3% average ATR)
- Flight to quality

**Neutral Regime:**
- Mixed performance (40-60% stocks positive)
- Average returns between -1% and +1%
- Sector rotation without clear direction
- Normal volume and volatility patterns

Provide your analysis in the following JSON format:
{{
    "regime": "<Risk-On/Risk-Off/Neutral>",
    "confidence": <0.0 to 1.0>,
    "dominant_themes": [<list of 3-5 key market themes>],
    "supporting_evidence": [<list of 3-5 data points supporting regime classification>],
    "sector_rotation": {{
        "outperforming": [<list of outperforming sectors>],
        "underperforming": [<list of underperforming sectors>]
    }},
    "portfolio_adjustments": {{
        "increase_exposure": [<list of recommended increases>],
        "decrease_exposure": [<list of recommended decreases>],
        "position_sizing": "<Conservative/Normal/Aggressive>"
    }},
    "key_risks": [<list of 2-3 key risks to monitor>],
    "outlook": "<1-2 sentence market outlook>"
}}

Focus on:
1. Clear regime classification based on the data
2. Actionable portfolio recommendations
3. Key themes driving the current environment
4. Risk factors to monitor

Respond ONLY with the JSON object."""

        # Get LLM response
        logger.info("Analyzing market regime...")
        response = self._call_ollama(prompt)
        
        if response:
            try:
                # Parse JSON response
                analysis = json.loads(response)
                
                # Validate and structure the response
                result = {
                    'date': date_str,
                    'regime': analysis.get('regime', 'Neutral'),
                    'confidence': float(analysis.get('confidence', 0.5)),
                    'dominant_themes': analysis.get('dominant_themes', [])[:5],
                    'supporting_evidence': analysis.get('supporting_evidence', [])[:5],
                    'sector_rotation': {
                        'outperforming': analysis.get('sector_rotation', {}).get('outperforming', [])[:3],
                        'underperforming': analysis.get('sector_rotation', {}).get('underperforming', [])[:3]
                    },
                    'portfolio_adjustments': {
                        'increase_exposure': analysis.get('portfolio_adjustments', {}).get('increase_exposure', [])[:3],
                        'decrease_exposure': analysis.get('portfolio_adjustments', {}).get('decrease_exposure', [])[:3],
                        'position_sizing': analysis.get('portfolio_adjustments', {}).get('position_sizing', 'Normal')
                    },
                    'key_risks': analysis.get('key_risks', [])[:3],
                    'outlook': analysis.get('outlook', 'Market direction unclear'),
                    'market_summary': market_summary,
                    'analysis_timestamp': datetime.now().isoformat()
                }
                
                # Save to cache
                self._save_to_cache(cache_key, result)
                
                # Also save to regime-specific cache
                regime_file = self.regime_cache_dir / f"regime_{date_str}.json"
                with open(regime_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Market regime classified as: {result['regime']} (confidence: {result['confidence']:.2f})")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse regime analysis: {e}")
                # Return fallback analysis
                return self._fallback_regime_analysis(market_summary)
        
        # Return fallback if LLM fails
        return self._fallback_regime_analysis(market_summary)
    
    def _fallback_regime_analysis(self, market_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Provide fallback regime analysis based on simple rules.
        
        Args:
            market_summary: Market data summary
            
        Returns:
            Basic regime analysis
        """
        perf = market_summary['market_performance']
        vol = market_summary['market_volatility']
        
        # Simple regime classification
        if perf['avg_return'] > 1.0 and perf['positive_stocks'] > perf['negative_stocks']:
            regime = 'Risk-On'
            confidence = 0.6
        elif perf['avg_return'] < -1.0 and perf['negative_stocks'] > perf['positive_stocks']:
            regime = 'Risk-Off'
            confidence = 0.6
        else:
            regime = 'Neutral'
            confidence = 0.4
        
        return {
            'date': market_summary['date'],
            'regime': regime,
            'confidence': confidence,
            'dominant_themes': ['Fallback analysis - LLM unavailable'],
            'supporting_evidence': [
                f"Average return: {perf['avg_return']:+.1f}%",
                f"Positive stocks: {perf['positive_stocks']}/{perf['positive_stocks'] + perf['negative_stocks']}",
                f"Average volatility: {vol['avg_volatility']:.1f}%"
            ],
            'sector_rotation': {
                'outperforming': [],
                'underperforming': []
            },
            'portfolio_adjustments': {
                'increase_exposure': [],
                'decrease_exposure': [],
                'position_sizing': 'Normal'
            },
            'key_risks': ['LLM analysis unavailable'],
            'outlook': 'Analysis limited due to LLM unavailability',
            'market_summary': market_summary,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def get_recent_regime_history(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get recent market regime history.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of regime analyses sorted by date
        """
        regimes = []
        
        for i in range(days_back):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            
            regime_file = self.regime_cache_dir / f"regime_{date_str}.json"
            if regime_file.exists():
                try:
                    with open(regime_file, 'r') as f:
                        regime_data = json.load(f)
                        regimes.append(regime_data)
                except Exception as e:
                    logger.error(f"Error loading regime data for {date_str}: {e}")
        
        # Sort by date (most recent first)
        regimes.sort(key=lambda x: x['date'], reverse=True)
        
        return regimes


def main():
    """Main function for standalone execution."""
    import argparse
    from data_collector import MarketDataCollector
    from news_collector import NewsCollector
    
    parser = argparse.ArgumentParser(description='LLM Stock Analyst')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to analyze')
    parser.add_argument('--test', action='store_true', help='Test LLM connection')
    parser.add_argument('--regime', action='store_true', help='Analyze market regime')
    parser.add_argument('--regime-history', action='store_true', help='Show recent regime history')
    
    args = parser.parse_args()
    
    # Initialize components
    analyst = LLMAnalyst()
    
    if args.test:
        # Test LLM connection
        print("Testing Ollama connection...")
        response = analyst._call_ollama("What is 2+2? Reply with just the number.")
        if response:
            print(f"✅ Ollama is working. Response: {response}")
        else:
            print("❌ Ollama connection failed")
        return
    
    if args.regime_history:
        # Show recent regime history
        regimes = analyst.get_recent_regime_history()
        if regimes:
            print("\n" + "="*80)
            print("RECENT MARKET REGIME HISTORY")
            print("="*80)
            for regime in regimes[:10]:  # Show last 10
                print(f"\n{regime['date']}: {regime['regime']} (confidence: {regime['confidence']:.2f})")
                print(f"Themes: {', '.join(regime['dominant_themes'][:3])}")
                print(f"Outlook: {regime['outlook']}")
        else:
            print("No regime history found.")
        return
    
    if args.regime:
        # Analyze current market regime
        print("Collecting market data for regime analysis...")
        market_collector = MarketDataCollector()
        
        # Get tickers for analysis
        tickers = args.tickers if args.tickers else MarketDataCollector.TECH_UNIVERSE[:15]  # Top 15
        
        # Collect market data
        stock_data = {}
        for ticker in tickers:
            df = market_collector.load_from_parquet(ticker)
            if df is not None and not df.empty:
                stock_data[ticker] = df
        
        if not stock_data:
            print("No market data available. Please run data collection first.")
            return
        
        # Generate market summary
        market_summary = analyst.get_market_summary(stock_data)
        
        # Analyze regime
        regime_analysis = analyst.analyze_market_regime(market_summary)
        
        # Display results
        print("\n" + "="*80)
        print("MARKET REGIME ANALYSIS")
        print("="*80)
        print(f"\nDate: {regime_analysis['date']}")
        print(f"Market Regime: {regime_analysis['regime']}")
        print(f"Confidence: {regime_analysis['confidence']:.2f}")
        
        print(f"\nDominant Themes:")
        for theme in regime_analysis['dominant_themes']:
            print(f"  • {theme}")
        
        print(f"\nSector Rotation:")
        if regime_analysis['sector_rotation']['outperforming']:
            print(f"  Outperforming: {', '.join(regime_analysis['sector_rotation']['outperforming'])}")
        if regime_analysis['sector_rotation']['underperforming']:
            print(f"  Underperforming: {', '.join(regime_analysis['sector_rotation']['underperforming'])}")
        
        print(f"\nPortfolio Adjustments:")
        adj = regime_analysis['portfolio_adjustments']
        print(f"  Position Sizing: {adj['position_sizing']}")
        if adj['increase_exposure']:
            print(f"  Increase Exposure: {', '.join(adj['increase_exposure'])}")
        if adj['decrease_exposure']:
            print(f"  Decrease Exposure: {', '.join(adj['decrease_exposure'])}")
        
        print(f"\nKey Risks:")
        for risk in regime_analysis['key_risks']:
            print(f"  • {risk}")
        
        print(f"\nOutlook: {regime_analysis['outlook']}")
        
        return
    
    # Get tickers
    tickers = args.tickers if args.tickers else ['AAPL', 'MSFT', 'NVDA']
    
    # Collect data
    print("Collecting market data and news...")
    market_collector = MarketDataCollector()
    news_collector = NewsCollector()
    
    stock_data = {}
    for ticker in tickers:
        # Get market data
        market_df = market_collector.load_from_parquet(ticker)
        if market_df is None or market_df.empty:
            print(f"No market data for {ticker}, skipping...")
            continue
        
        # Get news
        news = news_collector.get_weekly_news(ticker)
        if not news:
            news = news_collector.collect_stock_news(ticker)
        
        stock_data[ticker] = (news, market_df)
    
    # Analyze stocks
    print("\nAnalyzing stocks...")
    results = analyst.batch_analyze_stocks(stock_data)
    
    # Display results
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    
    for ticker, analysis in results.items():
        print(f"\n{ticker}:")
        print("-" * 40)
        print(f"Action: {analysis.get('action', 'N/A')}")
        print(f"Combined Score: {analysis.get('combined_score', 'N/A')}")
        print(f"Confidence: {analysis.get('confidence', 'N/A')}")
        print(f"Current Price: ${analysis.get('current_price', 0):.2f}")
        print(f"Support: ${analysis.get('support_level', 0):.2f}")
        print(f"Resistance: ${analysis.get('resistance_level', 0):.2f}")
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")


if __name__ == "__main__":
    main()