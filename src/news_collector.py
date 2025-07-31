#!/usr/bin/env python3
"""
News Collection Module
Collects news headlines for stocks from free sources
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsCollector:
    """Collects news headlines for stocks from various sources."""
    
    def __init__(self, data_dir: str = "data/news", cache_days: int = 7):
        """Initialize news collector.
        
        Args:
            data_dir: Directory to store news data
            cache_days: Number of days to cache news
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_days = cache_days
        self.rate_limit_delay = 1.0  # seconds between requests
        
        # User agent for web scraping
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        
    def _generate_news_id(self, ticker: str, headline: str, timestamp: str) -> str:
        """Generate unique ID for a news item."""
        content = f"{ticker}_{headline}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, date: datetime) -> Path:
        """Get cache file path for a specific date."""
        date_str = date.strftime('%Y-%m-%d')
        return self.data_dir / f"news_{date_str}.json"
    
    def load_cached_news(self, date: datetime) -> Dict[str, List[Dict]]:
        """Load cached news for a specific date.
        
        Args:
            date: Date to load news for
            
        Returns:
            Dictionary mapping ticker to list of news items
        """
        cache_path = self._get_cache_path(date)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache from {cache_path}: {e}")
        return {}
    
    def save_cached_news(self, date: datetime, news_data: Dict[str, List[Dict]]) -> None:
        """Save news data to cache.
        
        Args:
            date: Date of the news
            news_data: Dictionary mapping ticker to list of news items
        """
        cache_path = self._get_cache_path(date)
        try:
            with open(cache_path, 'w') as f:
                json.dump(news_data, f, indent=2, default=str)
            logger.info(f"Saved news cache to {cache_path}")
        except Exception as e:
            logger.error(f"Error saving cache to {cache_path}: {e}")
    
    def fetch_yahoo_finance_news(self, ticker: str) -> List[Dict]:
        """Fetch news from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news items
        """
        news_items = []
        
        try:
            # Use yfinance to get news
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if news:
                for item in news[:20]:  # Limit to 20 most recent
                    news_item = {
                        'ticker': ticker,
                        'headline': item.get('title', ''),
                        'source': item.get('publisher', 'Yahoo Finance'),
                        'url': item.get('link', ''),
                        'timestamp': datetime.fromtimestamp(
                            item.get('providerPublishTime', time.time())
                        ).isoformat(),
                        'summary': item.get('summary', '')
                    }
                    news_item['id'] = self._generate_news_id(
                        ticker, 
                        news_item['headline'], 
                        news_item['timestamp']
                    )
                    news_items.append(news_item)
                    
            logger.info(f"Fetched {len(news_items)} news items for {ticker} from Yahoo Finance")
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo Finance news for {ticker}: {e}")
            
        return news_items
    
    def fetch_finviz_news(self, ticker: str) -> List[Dict]:
        """Fetch news from Finviz using web scraping.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news items
        """
        news_items = []
        
        try:
            url = f"https://finviz.com/quote.ashx?t={ticker}"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                news_table = soup.find('table', {'class': 'fullview-news-outer'})
                
                if news_table:
                    rows = news_table.find_all('tr')
                    
                    for row in rows[:20]:  # Limit to 20 items
                        try:
                            link_cell = row.find('a', {'class': 'tab-link-news'})
                            if link_cell:
                                date_cell = row.find('td', {'align': 'right'})
                                
                                # Parse date/time
                                date_text = date_cell.text.strip() if date_cell else ''
                                
                                # Handle relative times
                                news_date = datetime.now()
                                if 'Today' in date_text:
                                    time_part = date_text.replace('Today', '').strip()
                                    # Keep today's date, just update time if provided
                                elif 'AM' in date_text or 'PM' in date_text:
                                    # Assume it's from today
                                    pass
                                else:
                                    # Try to parse the date
                                    try:
                                        news_date = datetime.strptime(date_text.split()[0], '%b-%d-%y')
                                    except:
                                        pass
                                
                                news_item = {
                                    'ticker': ticker,
                                    'headline': link_cell.text.strip(),
                                    'source': 'Finviz',
                                    'url': link_cell.get('href', ''),
                                    'timestamp': news_date.isoformat(),
                                    'summary': ''
                                }
                                news_item['id'] = self._generate_news_id(
                                    ticker,
                                    news_item['headline'],
                                    news_item['timestamp']
                                )
                                news_items.append(news_item)
                                
                        except Exception as e:
                            logger.debug(f"Error parsing Finviz news row: {e}")
                            
            logger.info(f"Fetched {len(news_items)} news items for {ticker} from Finviz")
            
        except Exception as e:
            logger.error(f"Error fetching Finviz news for {ticker}: {e}")
            
        return news_items
    
    def fetch_seeking_alpha_news(self, ticker: str) -> List[Dict]:
        """Fetch news from Seeking Alpha RSS feed.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            List of news items
        """
        news_items = []
        
        try:
            # Seeking Alpha RSS feed
            url = f"https://seekingalpha.com/api/sa/combined/{ticker}.xml"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'xml')
                items = soup.find_all('item')
                
                for item in items[:15]:  # Limit to 15 items
                    try:
                        title = item.find('title')
                        link = item.find('link')
                        pub_date = item.find('pubDate')
                        
                        if title and link:
                            # Parse publication date
                            news_date = datetime.now()
                            if pub_date:
                                try:
                                    from email.utils import parsedate_to_datetime
                                    news_date = parsedate_to_datetime(pub_date.text)
                                except:
                                    pass
                            
                            news_item = {
                                'ticker': ticker,
                                'headline': title.text.strip(),
                                'source': 'Seeking Alpha',
                                'url': link.text.strip(),
                                'timestamp': news_date.isoformat(),
                                'summary': ''
                            }
                            news_item['id'] = self._generate_news_id(
                                ticker,
                                news_item['headline'],
                                news_item['timestamp']
                            )
                            news_items.append(news_item)
                            
                    except Exception as e:
                        logger.debug(f"Error parsing Seeking Alpha item: {e}")
                        
            logger.info(f"Fetched {len(news_items)} news items for {ticker} from Seeking Alpha")
            
        except Exception as e:
            logger.error(f"Error fetching Seeking Alpha news for {ticker}: {e}")
            
        return news_items
    
    def collect_stock_news(self, ticker: str, use_cache: bool = True) -> List[Dict]:
        """Collect news for a single stock from all sources.
        
        Args:
            ticker: Stock ticker symbol
            use_cache: Whether to use cached data
            
        Returns:
            List of news items
        """
        today = datetime.now().date()
        
        # Check cache first
        if use_cache:
            cached_data = self.load_cached_news(datetime.now())
            if ticker in cached_data:
                logger.info(f"Using cached news for {ticker}")
                return cached_data[ticker]
        
        all_news = []
        
        # Fetch from Yahoo Finance
        yahoo_news = self.fetch_yahoo_finance_news(ticker)
        all_news.extend(yahoo_news)
        time.sleep(self.rate_limit_delay)
        
        # Fetch from Finviz
        finviz_news = self.fetch_finviz_news(ticker)
        all_news.extend(finviz_news)
        time.sleep(self.rate_limit_delay)
        
        # Fetch from Seeking Alpha
        sa_news = self.fetch_seeking_alpha_news(ticker)
        all_news.extend(sa_news)
        
        # Remove duplicates based on headline similarity
        unique_news = []
        seen_headlines = set()
        
        for item in all_news:
            # Create a normalized headline for comparison
            normalized = item['headline'].lower().strip()
            if normalized not in seen_headlines:
                seen_headlines.add(normalized)
                unique_news.append(item)
        
        # Sort by timestamp (newest first)
        unique_news.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Limit to 20 most recent
        unique_news = unique_news[:20]
        
        logger.info(f"Collected {len(unique_news)} unique news items for {ticker}")
        
        return unique_news
    
    def collect_multiple_stocks(self, tickers: List[str], use_cache: bool = True) -> Dict[str, List[Dict]]:
        """Collect news for multiple stocks.
        
        Args:
            tickers: List of stock ticker symbols
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping ticker to list of news items
        """
        all_news = {}
        total = len(tickers)
        
        for i, ticker in enumerate(tickers, 1):
            logger.info(f"Collecting news for {ticker} ({i}/{total})")
            
            try:
                news = self.collect_stock_news(ticker, use_cache)
                all_news[ticker] = news
                
                # Rate limiting between stocks
                if i < total:
                    time.sleep(self.rate_limit_delay)
                    
            except Exception as e:
                logger.error(f"Error collecting news for {ticker}: {e}")
                all_news[ticker] = []
        
        # Save to cache
        today = datetime.now()
        cached_data = self.load_cached_news(today)
        cached_data.update(all_news)
        self.save_cached_news(today, cached_data)
        
        return all_news
    
    def get_weekly_news(self, ticker: str, weeks_back: int = 1) -> List[Dict]:
        """Get news for a stock from the past N weeks.
        
        Args:
            ticker: Stock ticker symbol
            weeks_back: Number of weeks to look back
            
        Returns:
            List of news items from the past N weeks
        """
        all_news = []
        
        for days_ago in range(weeks_back * 7):
            date = datetime.now() - timedelta(days=days_ago)
            cached_data = self.load_cached_news(date)
            
            if ticker in cached_data:
                all_news.extend(cached_data[ticker])
        
        # Remove duplicates by ID
        unique_news = {}
        for item in all_news:
            unique_news[item['id']] = item
        
        # Sort by timestamp
        sorted_news = sorted(unique_news.values(), key=lambda x: x['timestamp'], reverse=True)
        
        return sorted_news
    
    def clean_old_cache(self) -> None:
        """Remove cache files older than cache_days."""
        cutoff_date = datetime.now() - timedelta(days=self.cache_days)
        
        for cache_file in self.data_dir.glob("news_*.json"):
            try:
                # Extract date from filename
                date_str = cache_file.stem.replace("news_", "")
                file_date = datetime.strptime(date_str, '%Y-%m-%d')
                
                if file_date < cutoff_date:
                    cache_file.unlink()
                    logger.info(f"Removed old cache file: {cache_file}")
                    
            except Exception as e:
                logger.error(f"Error cleaning cache file {cache_file}: {e}")


def main():
    """Main function for standalone execution."""
    import argparse
    from data_collector import MarketDataCollector
    
    parser = argparse.ArgumentParser(description='News Collector')
    parser.add_argument('--tickers', nargs='+', help='Specific tickers to collect news for')
    parser.add_argument('--no-cache', action='store_true', help='Skip cache and fetch fresh news')
    parser.add_argument('--show', help='Show news for a specific ticker')
    parser.add_argument('--clean', action='store_true', help='Clean old cache files')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = NewsCollector()
    
    if args.clean:
        collector.clean_old_cache()
        print("Cache cleaned.")
        return
    
    if args.show:
        # Show news for a specific ticker
        news = collector.get_weekly_news(args.show)
        print(f"\nNews for {args.show} (past week):")
        print("-" * 80)
        for item in news[:10]:
            print(f"\n[{item['timestamp'][:10]}] {item['source']}")
            print(f"{item['headline']}")
            print(f"URL: {item['url']}")
    else:
        # Collect news
        if args.tickers:
            tickers = args.tickers
        else:
            # Use default tech stocks
            tickers = MarketDataCollector.TECH_UNIVERSE[:10]  # Top 10 for demo
        
        use_cache = not args.no_cache
        results = collector.collect_multiple_stocks(tickers, use_cache=use_cache)
        
        # Print summary
        print("\nNews Collection Summary:")
        print("-" * 40)
        for ticker, news in results.items():
            print(f"{ticker}: {len(news)} headlines")


if __name__ == "__main__":
    main()