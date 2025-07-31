#!/usr/bin/env python3
"""
Market Data Collection Module
Collects and processes market data for tech stocks
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarketDataCollector:
    """Collects and processes market data for trading strategies."""
    
    # Define universe of 30 liquid tech stocks
    TECH_UNIVERSE = [
        # Mega-cap tech
        'AAPL',   # Apple
        'MSFT',   # Microsoft
        'GOOGL',  # Alphabet Class A
        'AMZN',   # Amazon
        'META',   # Meta Platforms
        'NVDA',   # NVIDIA
        'TSLA',   # Tesla
        
        # Large-cap tech
        'AVGO',   # Broadcom
        'ORCL',   # Oracle
        'ADBE',   # Adobe
        'CRM',    # Salesforce
        'AMD',    # Advanced Micro Devices
        'INTC',   # Intel
        'CSCO',   # Cisco
        'QCOM',   # Qualcomm
        
        # Software & Services
        'NFLX',   # Netflix
        'INTU',   # Intuit
        'NOW',    # ServiceNow
        'UBER',   # Uber
        'SHOP',   # Shopify
        'SQ',     # Block (Square)
        'PYPL',   # PayPal
        'ABNB',   # Airbnb
        
        # Semiconductors & Hardware
        'MU',     # Micron Technology
        'LRCX',   # Lam Research
        'AMAT',   # Applied Materials
        'MRVL',   # Marvell Technology
        
        # Other Tech
        'COIN',   # Coinbase
        'PLTR',   # Palantir
        'SNAP',   # Snap Inc
    ]
    
    def __init__(self, data_dir: str = "data/market"):
        """Initialize the data collector.
        
        Args:
            data_dir: Directory to store market data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.retry_delay = 2  # seconds
        self.max_retries = 3
        
    def download_stock_data(self, 
                          symbol: str, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Download stock data with retry logic.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Downloading data for {symbol} (attempt {attempt + 1}/{self.max_retries})")
                
                # Download data
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, auto_adjust=True)
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return None
                
                # Add symbol column
                df['Symbol'] = symbol
                
                return df
                
            except Exception as e:
                logger.error(f"Error downloading {symbol}: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Failed to download {symbol} after {self.max_retries} attempts")
                    
        return None
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period, min_periods=1).mean()
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=1).mean()
        
        return atr
    
    def calculate_volume_ratio(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """Calculate volume ratio (current volume / average volume)."""
        avg_volume = volume.rolling(window=period, min_periods=1).mean()
        return volume / avg_volume
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Simple Moving Averages
        df['SMA_20'] = self.calculate_sma(df['Close'], 20)
        df['SMA_50'] = self.calculate_sma(df['Close'], 50)
        
        # Price relative to SMAs
        df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
        df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        
        # RSI
        df['RSI_14'] = self.calculate_rsi(df['Close'], 14)
        
        # ATR
        df['ATR_14'] = self.calculate_atr(df, 14)
        
        # ATR Percentage (ATR as % of price)
        df['ATR_Percent'] = (df['ATR_14'] / df['Close']) * 100
        
        # Volume indicators
        df['Volume_Ratio'] = self.calculate_volume_ratio(df['Volume'], 20)
        df['Volume_SMA_20'] = self.calculate_sma(df['Volume'], 20)
        
        # Price changes
        df['Daily_Return'] = df['Close'].pct_change()
        df['Intraday_Range'] = (df['High'] - df['Low']) / df['Close']
        
        # Volatility (20-day rolling standard deviation of returns)
        df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252) * 100
        
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, symbol: str) -> None:
        """Save dataframe to parquet file.
        
        Args:
            df: DataFrame to save
            symbol: Stock symbol
        """
        filename = self.data_dir / f"{symbol}.parquet"
        df.to_parquet(filename, engine='pyarrow', compression='snappy')
        logger.info(f"Saved {symbol} data to {filename}")
    
    def load_from_parquet(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load dataframe from parquet file.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame or None if file doesn't exist
        """
        filename = self.data_dir / f"{symbol}.parquet"
        if filename.exists():
            return pd.read_parquet(filename, engine='pyarrow')
        return None
    
    def get_last_update_date(self, symbol: str) -> Optional[datetime]:
        """Get the last date in the saved data.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Last date or None if no data exists
        """
        df = self.load_from_parquet(symbol)
        if df is not None and len(df) > 0:
            return df.index[-1].to_pydatetime()
        return None
    
    def update_single_stock(self, symbol: str, force_full_download: bool = False) -> bool:
        """Update data for a single stock.
        
        Args:
            symbol: Stock symbol
            force_full_download: If True, download full history
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if force_full_download:
                # Download full history
                df = self.download_stock_data(symbol)
            else:
                # Check for existing data
                last_update = self.get_last_update_date(symbol)
                
                if last_update is None:
                    # No existing data, download full history
                    logger.info(f"No existing data for {symbol}, downloading full history")
                    df = self.download_stock_data(symbol)
                else:
                    # Download only new data
                    today = datetime.now()
                    days_since_update = (today - last_update).days
                    
                    if days_since_update <= 1:
                        logger.info(f"{symbol} is already up to date")
                        return True
                    
                    # Download from last update date
                    start_date = (last_update - timedelta(days=5)).strftime('%Y-%m-%d')
                    logger.info(f"Updating {symbol} from {start_date}")
                    
                    new_df = self.download_stock_data(symbol, start_date=start_date)
                    
                    if new_df is not None:
                        # Load existing data
                        existing_df = self.load_from_parquet(symbol)
                        
                        # Combine data (remove duplicates)
                        df = pd.concat([existing_df, new_df])
                        df = df[~df.index.duplicated(keep='last')]
                        df = df.sort_index()
                    else:
                        return False
            
            if df is not None and len(df) > 0:
                # Add technical indicators
                df = self.add_technical_indicators(df)
                
                # Save to parquet
                self.save_to_parquet(df, symbol)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating {symbol}: {str(e)}")
            return False
    
    def update_all_stocks(self, symbols: Optional[List[str]] = None, force_full_download: bool = False) -> Dict[str, bool]:
        """Update data for all stocks in the universe.
        
        Args:
            symbols: List of symbols to update (defaults to TECH_UNIVERSE)
            force_full_download: If True, download full history for all stocks
            
        Returns:
            Dictionary mapping symbol to success status
        """
        if symbols is None:
            symbols = self.TECH_UNIVERSE
            
        results = {}
        total = len(symbols)
        
        logger.info(f"Updating {total} stocks...")
        
        for i, symbol in enumerate(symbols, 1):
            logger.info(f"Processing {symbol} ({i}/{total})")
            success = self.update_single_stock(symbol, force_full_download)
            results[symbol] = success
            
            # Small delay to avoid rate limiting
            if i < total:
                time.sleep(0.5)
        
        # Summary
        successful = sum(results.values())
        logger.info(f"Update complete: {successful}/{total} stocks updated successfully")
        
        if successful < total:
            failed = [sym for sym, success in results.items() if not success]
            logger.warning(f"Failed to update: {', '.join(failed)}")
        
        return results
    
    def get_latest_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """Get the latest data for all symbols.
        
        Args:
            symbols: List of symbols (defaults to TECH_UNIVERSE)
            
        Returns:
            DataFrame with latest data for all symbols
        """
        if symbols is None:
            symbols = self.TECH_UNIVERSE
            
        all_data = []
        
        for symbol in symbols:
            df = self.load_from_parquet(symbol)
            if df is not None and len(df) > 0:
                # Get last row
                latest = df.iloc[-1:].copy()
                all_data.append(latest)
        
        if all_data:
            return pd.concat(all_data).sort_values('Symbol')
        else:
            return pd.DataFrame()
    
    def get_historical_data(self, 
                          symbol: str, 
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Get historical data for a symbol within a date range.
        
        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            DataFrame with filtered data or None
        """
        df = self.load_from_parquet(symbol)
        
        if df is None:
            return None
            
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
            
        return df


def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Market Data Collector')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to update')
    parser.add_argument('--full', action='store_true', help='Force full download')
    parser.add_argument('--summary', action='store_true', help='Show summary of latest data')
    
    args = parser.parse_args()
    
    # Initialize collector
    collector = MarketDataCollector()
    
    if args.summary:
        # Show summary of latest data
        latest = collector.get_latest_data(args.symbols)
        if not latest.empty:
            print("\nLatest Market Data Summary:")
            print("-" * 80)
            print(latest[['Symbol', 'Close', 'Volume', 'RSI_14', 'ATR_Percent', 'Volume_Ratio']])
        else:
            print("No data available. Run update first.")
    else:
        # Update data
        symbols = args.symbols if args.symbols else None
        results = collector.update_all_stocks(symbols, force_full_download=args.full)
        
        # Print summary
        print("\nUpdate Summary:")
        print("-" * 40)
        for symbol, success in results.items():
            status = "✓" if success else "✗"
            print(f"{status} {symbol}")


if __name__ == "__main__":
    main()