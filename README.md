# Autonomous Tech Stock Trading System

An AI-powered paper trading system for technology stocks using Ollama for decision-making and market analysis.

## Overview

This project implements an autonomous trading system that:
- Paper trades technology stocks with a $10,000 initial capital
- Uses AI (Ollama) for market analysis and trading decisions
- Supports up to 8 concurrent positions
- Implements risk management with stop-loss and take-profit orders
- Provides real-time monitoring through a web dashboard

## Features

- **AI-Driven Analysis**: Leverages Ollama LLM for market sentiment and technical analysis
- **Automated Trading**: Executes trades based on AI recommendations
- **Risk Management**: Built-in stop-loss, take-profit, and position sizing
- **Web Dashboard**: Real-time portfolio monitoring with Dash/Plotly
- **Backtesting**: Historical performance analysis capabilities
- **Configurable**: YAML-based configuration for easy customization

## Project Structure

```
stock-picker/
├── src/                 # Source code
├── data/               # Market data storage
├── logs/               # Application logs
├── tests/              # Unit and integration tests
├── config/             # Configuration files
│   └── config.yaml     # Main configuration
├── venv/               # Python virtual environment
├── .env.example        # Environment variables template
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup

1. **Clone the repository**
   ```bash
   cd stock-picker
   ```

2. **Create and activate virtual environment**
   ```bash
   # On macOS/Linux
   source venv/bin/activate
   
   # On Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. **Install Ollama**
   - Download from [ollama.ai](https://ollama.ai)
   - Pull the required model:
     ```bash
     ollama pull llama3.2
     ```

## Configuration

The system is configured through `config/config.yaml`:

- **Trading Parameters**:
  - Initial capital: $10,000
  - Maximum positions: 8
  - Rebalance frequency: Weekly
  
- **Risk Management**:
  - Stop loss: 5%
  - Take profit: 20%
  - Maximum drawdown: 15%

- **Stock Universe**: 16 major tech stocks including AAPL, MSFT, GOOGL, etc.

## Usage

### Running the Trading System

```bash
python src/main.py
```

### Launching the Dashboard

```bash
python src/dashboard.py
```

### Running Backtests

```bash
python src/backtest.py --start-date 2023-01-01 --end-date 2024-01-01
```

## Development

### Running Tests

```bash
pytest tests/
```

### Adding New Strategies

1. Create a new strategy class in `src/strategies/`
2. Implement the required interface methods
3. Register the strategy in the configuration

## API Keys

The system can optionally use:
- Broker APIs (Alpaca) for future real trading
- News APIs for sentiment analysis
- Webhook URLs for trade notifications

Configure these in your `.env` file.

## Logging

Logs are stored in the `logs/` directory with automatic rotation:
- Maximum file size: 10MB
- Backup count: 5 files
- Log levels: DEBUG, INFO, WARNING, ERROR

## Data Storage

- Market data is cached in the `data/` directory
- Cache expires after 1 hour by default
- Historical data spans 365 days

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## Disclaimer

This is a paper trading system for educational purposes. Always test thoroughly before using with real money. Past performance does not guarantee future results.

## License

[Your chosen license]