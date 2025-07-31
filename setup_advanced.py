#!/usr/bin/env python3
"""
Advanced Setup Script for Autonomous Trading System
Sets up Qwen 2.5 7B, LangChain, and infrastructure components
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import platform
import requests
import psutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('setup_advanced.log')
    ]
)
logger = logging.getLogger(__name__)


class SetupError(Exception):
    """Custom exception for setup errors."""
    pass


class AdvancedSetup:
    """Advanced setup manager for the trading system."""
    
    def __init__(self):
        """Initialize setup manager."""
        self.os_type = platform.system().lower()
        self.arch = platform.machine().lower()
        self.python_version = sys.version_info
        
        # System requirements
        self.min_ram_gb = 8
        self.min_disk_gb = 20
        
        # Component status
        self.status = {
            'system_check': False,
            'ollama_installed': False,
            'qwen_model': False,
            'langchain_deps': False,
            'docker_setup': False,
            'database_schema': False
        }
        
        logger.info(f"Setup initialized on {self.os_type} {self.arch}")
    
    def run_command(self, command: List[str], description: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a system command with logging.
        
        Args:
            command: Command to run as list
            description: Description for logging
            check: Whether to check return code
            
        Returns:
            CompletedProcess result
        """
        logger.info(f"Running: {description}")
        logger.debug(f"Command: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=check
            )
            
            if result.stdout:
                logger.debug(f"stdout: {result.stdout}")
            
            if result.stderr and result.returncode != 0:
                logger.error(f"stderr: {result.stderr}")
            
            return result
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {e}")
            logger.error(f"stdout: {e.stdout}")
            logger.error(f"stderr: {e.stderr}")
            raise SetupError(f"Failed to {description}")
        except FileNotFoundError:
            logger.error(f"Command not found: {command[0]}")
            raise SetupError(f"Command not found: {command[0]}")
    
    def check_system_requirements(self) -> bool:
        """Check system requirements."""
        logger.info("🔍 Checking system requirements...")
        
        try:
            # Check RAM
            ram_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"Available RAM: {ram_gb:.1f} GB")
            
            if ram_gb < self.min_ram_gb:
                logger.warning(f"RAM below recommended {self.min_ram_gb} GB")
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            logger.info(f"Available disk space: {free_gb:.1f} GB")
            
            if free_gb < self.min_disk_gb:
                logger.warning(f"Disk space below recommended {self.min_disk_gb} GB")
            
            # Check Python version
            if self.python_version < (3, 8):
                raise SetupError(f"Python 3.8+ required, found {sys.version}")
            
            logger.info(f"Python version: {sys.version}")
            
            # Check if running on macOS with Apple Silicon
            if self.os_type == 'darwin' and 'arm' in self.arch:
                logger.info("✅ Detected Apple Silicon Mac - optimal for Ollama")
            
            self.status['system_check'] = True
            logger.info("✅ System requirements check passed")
            return True
            
        except Exception as e:
            logger.error(f"System requirements check failed: {e}")
            return False
    
    def install_ollama(self) -> bool:
        """Install Ollama if not present."""
        logger.info("🚀 Installing Ollama...")
        
        try:
            # Check if Ollama is already installed
            try:
                result = self.run_command(['ollama', '--version'], "Check Ollama version", check=False)
                if result.returncode == 0:
                    logger.info("✅ Ollama already installed")
                    self.status['ollama_installed'] = True
                    return True
            except:
                pass
            
            # Install based on OS
            if self.os_type == 'darwin':  # macOS
                logger.info("Installing Ollama on macOS...")
                
                # Check if Homebrew is available
                try:
                    self.run_command(['brew', '--version'], "Check Homebrew", check=False)
                    # Install via Homebrew
                    self.run_command(['brew', 'install', 'ollama'], "Install Ollama via Homebrew")
                except:
                    # Install via curl script
                    logger.info("Homebrew not found, using curl installer...")
                    curl_command = [
                        'curl', '-fsSL', 'https://ollama.ai/install.sh'
                    ]
                    curl_result = self.run_command(curl_command, "Download Ollama installer")
                    
                    # Execute the installer
                    install_process = subprocess.Popen(
                        ['sh'],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = install_process.communicate(input=curl_result.stdout)
                    
                    if install_process.returncode != 0:
                        raise SetupError(f"Ollama installation failed: {stderr}")
            
            elif self.os_type == 'linux':
                logger.info("Installing Ollama on Linux...")
                curl_command = [
                    'curl', '-fsSL', 'https://ollama.ai/install.sh'
                ]
                curl_result = self.run_command(curl_command, "Download Ollama installer")
                
                # Execute installer with sudo
                install_process = subprocess.Popen(
                    ['sudo', 'sh'],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                stdout, stderr = install_process.communicate(input=curl_result.stdout)
                
                if install_process.returncode != 0:
                    raise SetupError(f"Ollama installation failed: {stderr}")
            
            else:
                raise SetupError(f"Unsupported OS: {self.os_type}")
            
            # Verify installation
            time.sleep(2)
            self.run_command(['ollama', '--version'], "Verify Ollama installation")
            
            self.status['ollama_installed'] = True
            logger.info("✅ Ollama installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Ollama installation failed: {e}")
            return False
    
    def setup_qwen_model(self) -> bool:
        """Download and test Qwen 2.5 7B model."""
        logger.info("🧠 Setting up Qwen 2.5 7B model...")
        
        try:
            # Start Ollama service if not running
            try:
                self.run_command(['ollama', 'list'], "Check Ollama service", check=False)
            except:
                logger.info("Starting Ollama service...")
                if self.os_type == 'darwin':
                    # On macOS, Ollama runs as a service
                    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(5)
                
            # Check if model already exists
            result = self.run_command(['ollama', 'list'], "List installed models", check=False)
            if 'qwen2.5:7b' in result.stdout:
                logger.info("✅ Qwen 2.5 7B already installed")
                self.status['qwen_model'] = True
                return self.test_qwen_model()
            
            # Pull Qwen 2.5 7B model
            logger.info("📥 Downloading Qwen 2.5 7B model (this may take several minutes)...")
            logger.info("💡 Model size: ~4.1GB - please be patient")
            
            pull_process = subprocess.Popen(
                ['ollama', 'pull', 'qwen2.5:7b'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Show progress
            for line in iter(pull_process.stdout.readline, ''):
                line = line.strip()
                if line:
                    if 'pulling' in line.lower() or '%' in line:
                        print(f"\r{line}", end='', flush=True)
                    else:
                        logger.info(line)
            
            pull_process.wait()
            print()  # New line after progress
            
            if pull_process.returncode != 0:
                raise SetupError("Failed to download Qwen 2.5 7B model")
            
            logger.info("✅ Qwen 2.5 7B model downloaded successfully")
            
            # Test the model
            return self.test_qwen_model()
            
        except Exception as e:
            logger.error(f"Qwen model setup failed: {e}")
            return False
    
    def test_qwen_model(self) -> bool:
        """Test Qwen model with complex reasoning task."""
        logger.info("🧪 Testing Qwen 2.5 7B with complex reasoning...")
        
        try:
            # Complex reasoning test prompt
            test_prompt = """
            You are analyzing a tech stock for investment. Given these metrics:
            - P/E ratio: 28.5
            - Revenue growth: 15% YoY
            - Profit margin: 22%
            - Debt-to-equity: 0.3
            - Market cap: $500B
            - Recent news: Strong earnings beat, new product launch
            
            Provide a structured analysis with:
            1. Valuation assessment (overvalued/fair/undervalued)
            2. Growth prospects (high/medium/low)
            3. Risk factors (list top 3)
            4. Investment recommendation (buy/hold/sell)
            5. Confidence level (1-10)
            
            Format as JSON.
            """
            
            # Create temporary file for prompt
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(test_prompt)
                temp_file = f.name
            
            try:
                # Run Ollama with the test prompt
                result = self.run_command(
                    ['ollama', 'run', 'qwen2.5:7b'],
                    "Test Qwen model reasoning",
                    check=False
                )
                
                # Alternative: Use echo and pipe
                echo_process = subprocess.Popen(
                    ['echo', test_prompt],
                    stdout=subprocess.PIPE
                )
                
                ollama_process = subprocess.Popen(
                    ['ollama', 'run', 'qwen2.5:7b'],
                    stdin=echo_process.stdout,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                echo_process.stdout.close()
                stdout, stderr = ollama_process.communicate(timeout=60)
                
                if ollama_process.returncode == 0 and stdout.strip():
                    logger.info("✅ Qwen model test successful")
                    logger.info(f"Sample response length: {len(stdout)} characters")
                    
                    # Log first 200 characters of response
                    sample = stdout[:200].replace('\n', ' ')
                    logger.info(f"Sample response: {sample}...")
                    
                    self.status['qwen_model'] = True
                    return True
                else:
                    logger.error(f"Model test failed: {stderr}")
                    return False
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
            
        except subprocess.TimeoutExpired:
            logger.error("Model test timed out")
            return False
        except Exception as e:
            logger.error(f"Model test failed: {e}")
            return False
    
    def install_langchain_dependencies(self) -> bool:
        """Install LangChain and related dependencies."""
        logger.info("📦 Installing LangChain dependencies...")
        
        try:
            # Define packages to install
            packages = [
                'langchain',
                'langchain-community',
                'langchain-ollama',
                'chromadb',
                'celery',
                'redis',
                'sqlalchemy',
                'psycopg2-binary',
                'gradio',
                'instructor',
                'pydantic>=2.0',
                'fastapi',
                'uvicorn',
                'python-multipart',
                'jinja2',
                'python-dotenv',
                'beautifulsoup4',
                'lxml',
                'aiohttp',
                'asyncpg'
            ]
            
            # Install packages
            for package in packages:
                logger.info(f"Installing {package}...")
                self.run_command(
                    [sys.executable, '-m', 'pip', 'install', package],
                    f"Install {package}"
                )
            
            # Verify key imports
            test_imports = [
                'langchain',
                'langchain_community',
                'chromadb',
                'celery',
                'redis',
                'sqlalchemy',
                'gradio',
                'instructor',
                'pydantic'
            ]
            
            logger.info("🔍 Verifying imports...")
            for module in test_imports:
                try:
                    __import__(module)
                    logger.debug(f"✅ {module} imported successfully")
                except ImportError as e:
                    logger.error(f"❌ Failed to import {module}: {e}")
                    return False
            
            self.status['langchain_deps'] = True
            logger.info("✅ LangChain dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"LangChain dependencies installation failed: {e}")
            return False
    
    def create_docker_compose(self) -> bool:
        """Create Docker Compose configuration."""
        logger.info("🐳 Creating Docker Compose configuration...")
        
        try:
            docker_compose_content = """version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: trading_postgres
    environment:
      POSTGRES_DB: trading_system
      POSTGRES_USER: trader
      POSTGRES_PASSWORD: trading_secure_2024
      POSTGRES_INITDB_ARGS: "--encoding=UTF-8 --lc-collate=C --lc-ctype=C"
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    networks:
      - trading_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U trader -d trading_system"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # Redis for Celery task queue
  redis:
    image: redis:7-alpine
    container_name: trading_redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - trading_network
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  # pgAdmin for database management
  pgadmin:
    image: dpage/pgadmin4:latest
    container_name: trading_pgadmin
    environment:
      PGADMIN_DEFAULT_EMAIL: admin@trading.local
      PGADMIN_DEFAULT_PASSWORD: admin_secure_2024
      PGADMIN_CONFIG_SERVER_MODE: 'False'
    ports:
      - "8080:80"
    volumes:
      - pgadmin_data:/var/lib/pgadmin
    networks:
      - trading_network
    depends_on:
      postgres:
        condition: service_healthy
    restart: unless-stopped

  # Celery Worker
  celery_worker:
    build: 
      context: .
      dockerfile: Dockerfile.celery
    container_name: trading_celery_worker
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://trader:trading_secure_2024@postgres:5432/trading_system
    volumes:
      - ./src:/app/src
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - trading_network
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    command: celery -A src.tasks worker --loglevel=info

  # Celery Beat (Scheduler)
  celery_beat:
    build: 
      context: .
      dockerfile: Dockerfile.celery
    container_name: trading_celery_beat
    environment:
      - CELERY_BROKER_URL=redis://redis:6379/0
      - DATABASE_URL=postgresql://trader:trading_secure_2024@postgres:5432/trading_system
    volumes:
      - ./src:/app/src
      - ./data:/app/data
      - ./logs:/app/logs
    networks:
      - trading_network
    depends_on:
      - redis
      - postgres
    restart: unless-stopped
    command: celery -A src.tasks beat --loglevel=info

volumes:
  postgres_data:
    driver: local
  redis_data:
    driver: local
  pgadmin_data:
    driver: local

networks:
  trading_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
"""
            
            # Write docker-compose.yml
            with open('docker-compose.yml', 'w') as f:
                f.write(docker_compose_content)
            
            # Create Dockerfile for Celery
            dockerfile_content = """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-m", "celery", "worker", "-A", "src.tasks"]
"""
            
            with open('Dockerfile.celery', 'w') as f:
                f.write(dockerfile_content)
            
            # Create environment file template
            env_content = """# Database Configuration
DATABASE_URL=postgresql://trader:trading_secure_2024@localhost:5432/trading_system

# Redis Configuration
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/0

# Email Configuration (for notifications)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
NOTIFICATION_EMAILS=admin@example.com

# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# API Keys (if needed)
ALPHA_VANTAGE_API_KEY=your-api-key
POLYGON_API_KEY=your-api-key

# System Configuration
LOG_LEVEL=INFO
DEBUG=False
ENVIRONMENT=development
"""
            
            with open('.env.example', 'w') as f:
                f.write(env_content)
            
            # Create init scripts directory
            init_scripts_dir = Path('init-scripts')
            init_scripts_dir.mkdir(exist_ok=True)
            
            self.status['docker_setup'] = True
            logger.info("✅ Docker Compose configuration created")
            return True
            
        except Exception as e:
            logger.error(f"Docker Compose setup failed: {e}")
            return False
    
    def create_database_schema(self) -> bool:
        """Create database schema initialization script."""
        logger.info("🗄️ Creating database schema...")
        
        try:
            schema_sql = """-- Autonomous Trading System Database Schema
-- Generated by setup_advanced.py

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Decisions table - stores all trading decisions
CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    type VARCHAR(50) NOT NULL, -- 'rebalance', 'risk_management', 'manual'
    details JSONB NOT NULL,
    reasoning TEXT,
    confidence_score DECIMAL(5,4), -- 0.0000 to 1.0000
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'approved', 'rejected', 'executed'
    approved_by VARCHAR(100),
    executed_at TIMESTAMP WITH TIME ZONE,
    created_by VARCHAR(100) DEFAULT 'system',
    
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'approved', 'rejected', 'executed', 'cancelled'))
);

-- Analysis table - stores LLM and quantitative analysis
CREATE TABLE IF NOT EXISTS analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    stock VARCHAR(10) NOT NULL,
    analysis_type VARCHAR(50) NOT NULL, -- 'sentiment', 'technical', 'fundamental', 'risk'
    metrics JSONB NOT NULL,
    llm_output TEXT,
    llm_model VARCHAR(50) DEFAULT 'qwen2.5:7b',
    confidence_score DECIMAL(5,4),
    data_sources TEXT[], -- array of data sources used
    processing_time_ms INTEGER,
    
    CONSTRAINT valid_stock_symbol CHECK (LENGTH(stock) <= 10),
    CONSTRAINT valid_confidence CHECK (confidence_score >= 0 AND confidence_score <= 1)
);

-- Trades table - stores all trade executions
CREATE TABLE IF NOT EXISTS trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    decision_id UUID REFERENCES decisions(id),
    action VARCHAR(10) NOT NULL, -- 'BUY', 'SELL'
    stock VARCHAR(10) NOT NULL,
    quantity DECIMAL(12,4) NOT NULL,
    price DECIMAL(12,4) NOT NULL,
    total_value DECIMAL(15,2) GENERATED ALWAYS AS (quantity * price) STORED,
    transaction_cost DECIMAL(8,2) DEFAULT 0,
    slippage DECIMAL(8,4) DEFAULT 0,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'filled', 'partial', 'cancelled', 'failed'
    exchange VARCHAR(20) DEFAULT 'paper', -- 'paper', 'nasdaq', 'nyse'
    order_type VARCHAR(20) DEFAULT 'market', -- 'market', 'limit', 'stop'
    filled_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT valid_action CHECK (action IN ('BUY', 'SELL')),
    CONSTRAINT valid_quantity CHECK (quantity > 0),
    CONSTRAINT valid_price CHECK (price > 0),
    CONSTRAINT valid_status CHECK (status IN ('pending', 'filled', 'partial', 'cancelled', 'failed'))
);

-- Conversations table - stores LLM conversations and context
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    session_id UUID,
    query TEXT NOT NULL,
    response TEXT NOT NULL,
    context JSONB,
    model VARCHAR(50) DEFAULT 'qwen2.5:7b',
    tokens_used INTEGER,
    response_time_ms INTEGER,
    user_id VARCHAR(100) DEFAULT 'system',
    conversation_type VARCHAR(50), -- 'analysis', 'decision', 'support', 'research'
    
    CONSTRAINT valid_tokens CHECK (tokens_used >= 0),
    CONSTRAINT valid_response_time CHECK (response_time_ms >= 0)
);

-- System logs table - comprehensive logging
CREATE TABLE IF NOT EXISTS system_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    component VARCHAR(100) NOT NULL, -- 'data_collector', 'signal_aggregator', 'portfolio_manager', etc.
    action VARCHAR(100) NOT NULL,
    level VARCHAR(10) DEFAULT 'INFO', -- 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
    message TEXT NOT NULL,
    details JSONB,
    execution_time_ms INTEGER,
    error_code VARCHAR(50),
    stack_trace TEXT,
    
    CONSTRAINT valid_level CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'))
);

-- Performance tracking table
CREATE TABLE IF NOT EXISTS performance_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    portfolio_value DECIMAL(15,2) NOT NULL,
    cash_balance DECIMAL(15,2) NOT NULL,
    positions JSONB NOT NULL,
    daily_return DECIMAL(8,6),
    cumulative_return DECIMAL(8,6),
    benchmark_return DECIMAL(8,6),
    sharpe_ratio DECIMAL(8,4),
    max_drawdown DECIMAL(8,6),
    volatility DECIMAL(8,6),
    beta DECIMAL(6,4),
    alpha DECIMAL(8,6),
    
    CONSTRAINT valid_portfolio_value CHECK (portfolio_value >= 0)
);

-- Market data cache table
CREATE TABLE IF NOT EXISTS market_data_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(12,4),
    high_price DECIMAL(12,4),
    low_price DECIMAL(12,4),
    close_price DECIMAL(12,4),
    volume BIGINT,
    adjusted_close DECIMAL(12,4),
    technical_indicators JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(symbol, date),
    CONSTRAINT valid_prices CHECK (
        open_price > 0 AND high_price > 0 AND 
        low_price > 0 AND close_price > 0 AND
        high_price >= low_price
    )
);

-- News data table
CREATE TABLE IF NOT EXISTS news_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(10),
    headline TEXT NOT NULL,
    summary TEXT,
    url TEXT,
    source VARCHAR(100),
    published_at TIMESTAMP WITH TIME ZONE,
    sentiment_score DECIMAL(5,4), -- -1.0000 to 1.0000
    sentiment_label VARCHAR(20), -- 'positive', 'negative', 'neutral'
    relevance_score DECIMAL(5,4), -- 0.0000 to 1.0000
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT valid_sentiment CHECK (sentiment_score >= -1 AND sentiment_score <= 1),
    CONSTRAINT valid_relevance CHECK (relevance_score >= 0 AND relevance_score <= 1)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_type ON decisions(type);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON decisions(status);

CREATE INDEX IF NOT EXISTS idx_analysis_timestamp ON analysis(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_stock ON analysis(stock);
CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis(analysis_type);

CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_stock ON trades(stock);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
CREATE INDEX IF NOT EXISTS idx_trades_decision_id ON trades(decision_id);

CREATE INDEX IF NOT EXISTS idx_conversations_timestamp ON conversations(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_type ON conversations(conversation_type);

CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_logs_component ON system_logs(component);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);

CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_snapshots(timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date ON market_data_cache(symbol, date DESC);

CREATE INDEX IF NOT EXISTS idx_news_symbol ON news_data(symbol);
CREATE INDEX IF NOT EXISTS idx_news_published ON news_data(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_data(sentiment_score);

-- Create GIN indexes for JSONB columns
CREATE INDEX IF NOT EXISTS idx_decisions_details_gin ON decisions USING GIN(details);
CREATE INDEX IF NOT EXISTS idx_analysis_metrics_gin ON analysis USING GIN(metrics);
CREATE INDEX IF NOT EXISTS idx_conversations_context_gin ON conversations USING GIN(context);
CREATE INDEX IF NOT EXISTS idx_system_logs_details_gin ON system_logs USING GIN(details);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_conversations_query_fts ON conversations USING GIN(to_tsvector('english', query));
CREATE INDEX IF NOT EXISTS idx_conversations_response_fts ON conversations USING GIN(to_tsvector('english', response));
CREATE INDEX IF NOT EXISTS idx_news_headline_fts ON news_data USING GIN(to_tsvector('english', headline));

-- Create views for common queries
CREATE OR REPLACE VIEW recent_decisions AS
SELECT 
    id,
    timestamp,
    type,
    details->>'summary' as summary,
    confidence_score,
    status,
    approved_by
FROM decisions 
WHERE timestamp >= NOW() - INTERVAL '30 days'
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
    timestamp,
    portfolio_value,
    cash_balance,
    daily_return,
    cumulative_return,
    sharpe_ratio,
    max_drawdown
FROM performance_snapshots 
WHERE timestamp >= NOW() - INTERVAL '90 days'
ORDER BY timestamp DESC;

CREATE OR REPLACE VIEW trade_summary AS
SELECT 
    DATE(timestamp) as trade_date,
    COUNT(*) as total_trades,
    SUM(CASE WHEN action = 'BUY' THEN 1 ELSE 0 END) as buy_trades,
    SUM(CASE WHEN action = 'SELL' THEN 1 ELSE 0 END) as sell_trades,
    SUM(total_value) as total_volume,
    AVG(transaction_cost) as avg_transaction_cost
FROM trades
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(timestamp)
ORDER BY trade_date DESC;

-- Insert initial system configuration
INSERT INTO system_logs (component, action, level, message, details) VALUES 
    ('database', 'schema_initialization', 'INFO', 'Database schema initialized successfully', 
     '{"version": "1.0", "tables_created": 8, "indexes_created": 15, "views_created": 3}')
ON CONFLICT DO NOTHING;

-- Create functions for common operations
CREATE OR REPLACE FUNCTION log_system_event(
    p_component VARCHAR(100),
    p_action VARCHAR(100),
    p_level VARCHAR(10),
    p_message TEXT,
    p_details JSONB DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    log_id UUID;
BEGIN
    INSERT INTO system_logs (component, action, level, message, details)
    VALUES (p_component, p_action, p_level, p_message, p_details)
    RETURNING id INTO log_id;
    
    RETURN log_id;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION get_portfolio_performance(days_back INTEGER DEFAULT 30)
RETURNS TABLE(
    date DATE,
    portfolio_value DECIMAL(15,2),
    daily_return DECIMAL(8,6),
    cumulative_return DECIMAL(8,6)
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ps.timestamp::DATE,
        ps.portfolio_value,
        ps.daily_return,
        ps.cumulative_return
    FROM performance_snapshots ps
    WHERE ps.timestamp >= NOW() - (days_back || ' days')::INTERVAL
    ORDER BY ps.timestamp;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trader;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trader;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO trader;

-- Final log entry
SELECT log_system_event(
    'database', 
    'schema_complete', 
    'INFO', 
    'Database schema setup completed successfully',
    '{"timestamp": "' || NOW() || '", "status": "ready"}'::jsonb
);
"""
            
            # Write schema to init script
            init_scripts_dir = Path('init-scripts')
            init_scripts_dir.mkdir(exist_ok=True)
            
            with open(init_scripts_dir / '01-schema.sql', 'w') as f:
                f.write(schema_sql)
            
            # Create Python database utilities
            db_utils_content = '''"""
Database utilities for the autonomous trading system.
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import uuid
import json

import asyncpg
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Async database manager for the trading system."""
    
    def __init__(self, database_url: str):
        """Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL
        """
        self.database_url = database_url
        self.engine = create_async_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def log_decision(self, 
                          decision_type: str,
                          details: Dict[str, Any],
                          reasoning: str,
                          confidence_score: float = None) -> str:
        """Log a trading decision.
        
        Args:
            decision_type: Type of decision
            details: Decision details as JSON
            reasoning: Human-readable reasoning
            confidence_score: Confidence level (0-1)
            
        Returns:
            Decision ID
        """
        async with self.SessionLocal() as session:
            query = """
                INSERT INTO decisions (type, details, reasoning, confidence_score)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """
            
            result = await session.execute(
                sa.text(query),
                {
                    'type': decision_type,
                    'details': json.dumps(details),
                    'reasoning': reasoning,
                    'confidence_score': confidence_score
                }
            )
            
            decision_id = result.fetchone()[0]
            await session.commit()
            
            return str(decision_id)
    
    async def log_analysis(self,
                          stock: str,
                          analysis_type: str,
                          metrics: Dict[str, Any],
                          llm_output: str = None,
                          confidence_score: float = None) -> str:
        """Log analysis results.
        
        Args:
            stock: Stock symbol
            analysis_type: Type of analysis
            metrics: Analysis metrics
            llm_output: LLM output text
            confidence_score: Confidence level
            
        Returns:
            Analysis ID
        """
        async with self.SessionLocal() as session:
            query = """
                INSERT INTO analysis (stock, analysis_type, metrics, llm_output, confidence_score)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            
            result = await session.execute(
                sa.text(query),
                {
                    'stock': stock,
                    'analysis_type': analysis_type,
                    'metrics': json.dumps(metrics),
                    'llm_output': llm_output,
                    'confidence_score': confidence_score
                }
            )
            
            analysis_id = result.fetchone()[0]
            await session.commit()
            
            return str(analysis_id)
    
    async def log_trade(self,
                       action: str,
                       stock: str,
                       quantity: float,
                       price: float,
                       decision_id: str = None) -> str:
        """Log a trade execution.
        
        Args:
            action: BUY or SELL
            stock: Stock symbol
            quantity: Number of shares
            price: Price per share
            decision_id: Related decision ID
            
        Returns:
            Trade ID
        """
        async with self.SessionLocal() as session:
            query = """
                INSERT INTO trades (action, stock, quantity, price, decision_id)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id
            """
            
            result = await session.execute(
                sa.text(query),
                {
                    'action': action,
                    'stock': stock,
                    'quantity': quantity,
                    'price': price,
                    'decision_id': decision_id
                }
            )
            
            trade_id = result.fetchone()[0]
            await session.commit()
            
            return str(trade_id)
    
    async def log_conversation(self,
                              query: str,
                              response: str,
                              context: Dict[str, Any] = None,
                              session_id: str = None) -> str:
        """Log LLM conversation.
        
        Args:
            query: User query
            response: LLM response
            context: Conversation context
            session_id: Session identifier
            
        Returns:
            Conversation ID
        """
        async with self.SessionLocal() as session:
            db_query = """
                INSERT INTO conversations (query, response, context, session_id)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """
            
            result = await session.execute(
                sa.text(db_query),
                {
                    'query': query,
                    'response': response,
                    'context': json.dumps(context) if context else None,
                    'session_id': session_id
                }
            )
            
            conversation_id = result.fetchone()[0]
            await session.commit()
            
            return str(conversation_id)
    
    async def log_system_event(self,
                              component: str,
                              action: str,
                              message: str,
                              level: str = 'INFO',
                              details: Dict[str, Any] = None) -> str:
        """Log system event.
        
        Args:
            component: System component name
            action: Action performed
            message: Log message
            level: Log level
            details: Additional details
            
        Returns:
            Log entry ID
        """
        async with self.SessionLocal() as session:
            query = """
                SELECT log_system_event($1, $2, $3, $4, $5)
            """
            
            result = await session.execute(
                sa.text(query),
                {
                    'component': component,
                    'action': action,
                    'level': level,
                    'message': message,
                    'details': json.dumps(details) if details else None
                }
            )
            
            log_id = result.fetchone()[0]
            await session.commit()
            
            return str(log_id)


# Database connection helper
def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://trader:trading_secure_2024@localhost:5432/trading_system')
    return DatabaseManager(database_url)


# Usage example
async def example_usage():
    """Example of how to use the database manager."""
    db = get_database_manager()
    
    # Log a decision
    decision_id = await db.log_decision(
        decision_type='rebalance',
        details={'target_weights': {'AAPL': 0.2, 'MSFT': 0.15}},
        reasoning='Weekly rebalancing based on signal analysis',
        confidence_score=0.85
    )
    
    # Log analysis
    analysis_id = await db.log_analysis(
        stock='AAPL',
        analysis_type='sentiment',
        metrics={'sentiment_score': 0.7, 'confidence': 0.8},
        llm_output='Positive sentiment based on recent earnings...'
    )
    
    # Log trade
    trade_id = await db.log_trade(
        action='BUY',
        stock='AAPL',
        quantity=100,
        price=150.50,
        decision_id=decision_id
    )
    
    print(f"Logged decision: {decision_id}")
    print(f"Logged analysis: {analysis_id}")
    print(f"Logged trade: {trade_id}")


if __name__ == "__main__":
    asyncio.run(example_usage())
'''
            
            with open('src/database_utils.py', 'w') as f:
                f.write(db_utils_content)
            
            self.status['database_schema'] = True
            logger.info("✅ Database schema created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database schema creation failed: {e}")
            return False
    
    def create_test_scripts(self) -> bool:
        """Create test scripts to verify setup."""
        logger.info("🧪 Creating test scripts...")
        
        try:
            # Test Ollama connection
            test_ollama_content = '''#!/usr/bin/env python3
"""
Test Ollama and Qwen 2.5 7B setup
"""

import requests
import json
import time

def test_ollama_connection():
    """Test basic Ollama connection."""
    try:
        response = requests.get('http://localhost:11434/api/version', timeout=5)
        if response.status_code == 200:
            print("✅ Ollama service is running")
            return True
        else:
            print(f"❌ Ollama service responded with status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to Ollama service")
        return False
    except Exception as e:
        print(f"❌ Error testing Ollama connection: {e}")
        return False

def test_qwen_model():
    """Test Qwen 2.5 7B model."""
    try:
        prompt = "Analyze this stock scenario: AAPL trading at $150, P/E ratio 25, growing 10% YoY. Give a brief buy/hold/sell recommendation."
        
        payload = {
            "model": "qwen2.5:7b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 200
            }
        }
        
        print("Testing Qwen 2.5 7B model...")
        start_time = time.time()
        
        response = requests.post(
            'http://localhost:11434/api/generate',
            json=payload,
            timeout=60
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            response_text = result.get('response', '')
            
            print(f"✅ Qwen model test successful")
            print(f"Response time: {end_time - start_time:.2f} seconds")
            print(f"Response length: {len(response_text)} characters")
            print(f"Sample response: {response_text[:200]}...")
            return True
        else:
            print(f"❌ Model test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Qwen model: {e}")
        return False

def main():
    print("🧪 Testing Ollama and Qwen 2.5 7B Setup")
    print("=" * 50)
    
    # Test connection
    if not test_ollama_connection():
        print("\\nPlease start Ollama service:")
        print("- macOS: ollama serve")
        print("- Linux: systemctl start ollama")
        return
    
    # Test model
    if not test_qwen_model():
        print("\\nPlease ensure Qwen 2.5 7B is installed:")
        print("- ollama pull qwen2.5:7b")
        return
    
    print("\\n🎉 All tests passed! Ollama and Qwen 2.5 7B are ready.")

if __name__ == "__main__":
    main()
'''
            
            with open('test_ollama.py', 'w') as f:
                f.write(test_ollama_content)
            
            # Test database connection
            test_db_content = '''#!/usr/bin/env python3
"""
Test database connection and schema
"""

import asyncio
import os
import sys
sys.path.append('src')

try:
    import asyncpg
    import sqlalchemy as sa
    from database_utils import get_database_manager
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install dependencies: pip install asyncpg sqlalchemy")
    sys.exit(1)

async def test_database_connection():
    """Test database connection."""
    try:
        database_url = os.getenv('DATABASE_URL', 'postgresql://trader:trading_secure_2024@localhost:5432/trading_system')
        
        # Test basic connection
        conn = await asyncpg.connect(database_url)
        await conn.close()
        
        print("✅ Database connection successful")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

async def test_database_schema():
    """Test database schema."""
    try:
        db = get_database_manager()
        
        # Test logging functions
        decision_id = await db.log_decision(
            decision_type='test',
            details={'test': True},
            reasoning='Test decision',
            confidence_score=0.5
        )
        
        analysis_id = await db.log_analysis(
            stock='TEST',
            analysis_type='test',
            metrics={'test_metric': 1.0},
            llm_output='Test analysis output'
        )
        
        trade_id = await db.log_trade(
            action='BUY',
            stock='TEST',
            quantity=1.0,
            price=100.0,
            decision_id=decision_id
        )
        
        conversation_id = await db.log_conversation(
            query='Test query',
            response='Test response',
            context={'test': True}
        )
        
        log_id = await db.log_system_event(
            component='test',
            action='test_setup',
            message='Test setup verification',
            details={'test': True}
        )
        
        print("✅ Database schema test successful")
        print(f"Created test records:")
        print(f"  Decision: {decision_id}")
        print(f"  Analysis: {analysis_id}")
        print(f"  Trade: {trade_id}")
        print(f"  Conversation: {conversation_id}")
        print(f"  Log: {log_id}")
        
        return True
        
    except Exception as e:
        print(f"❌ Database schema test failed: {e}")
        return False

async def main():
    print("🧪 Testing Database Setup")
    print("=" * 50)
    
    # Test connection
    if not await test_database_connection():
        print("\\nPlease start PostgreSQL and create database:")
        print("- docker-compose up -d postgres")
        return
    
    # Test schema
    if not await test_database_schema():
        print("\\nPlease run schema initialization:")
        print("- docker-compose up -d postgres")
        print("- Wait for database to initialize")
        return
    
    print("\\n🎉 Database setup test passed!")

if __name__ == "__main__":
    asyncio.run(main())
'''
            
            with open('test_database.py', 'w') as f:
                f.write(test_db_content)
            
            # Test LangChain setup
            test_langchain_content = '''#!/usr/bin/env python3
"""
Test LangChain setup and integration
"""

import sys

def test_imports():
    """Test LangChain imports."""
    try:
        import langchain
        import langchain_community
        import chromadb
        import celery
        import redis
        import gradio
        import instructor
        import pydantic
        
        print("✅ All LangChain dependencies imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_langchain_ollama():
    """Test LangChain with Ollama."""
    try:
        from langchain_community.llms import Ollama
        
        llm = Ollama(
            model="qwen2.5:7b",
            base_url="http://localhost:11434"
        )
        
        # Test simple query
        response = llm.invoke("What is 2+2? Answer briefly.")
        
        print("✅ LangChain + Ollama integration successful")
        print(f"Sample response: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"❌ LangChain + Ollama test failed: {e}")
        return False

def test_vector_store():
    """Test ChromaDB vector store."""
    try:
        import chromadb
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import OllamaEmbeddings
        
        # Test ChromaDB
        client = chromadb.Client()
        collection = client.create_collection("test")
        
        # Add test documents
        collection.add(
            documents=["Apple is a technology company", "Microsoft makes software"],
            ids=["1", "2"]
        )
        
        results = collection.query(query_texts=["technology"], n_results=1)
        
        print("✅ ChromaDB vector store test successful")
        print(f"Query results: {len(results['documents'][0])} documents")
        return True
        
    except Exception as e:
        print(f"❌ Vector store test failed: {e}")
        return False

def main():
    print("🧪 Testing LangChain Setup")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("LangChain + Ollama", test_langchain_ollama),
        ("Vector Store", test_vector_store)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\\nRunning {test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 LangChain setup is complete and functional!")
    else:
        print("⚠️ Some tests failed. Check dependencies and services.")

if __name__ == "__main__":
    main()
'''
            
            with open('test_langchain.py', 'w') as f:
                f.write(test_langchain_content)
            
            # Make test scripts executable
            os.chmod('test_ollama.py', 0o755)
            os.chmod('test_database.py', 0o755)
            os.chmod('test_langchain.py', 0o755)
            
            logger.info("✅ Test scripts created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Test script creation failed: {e}")
            return False
    
    def run_full_setup(self) -> bool:
        """Run the complete setup process."""
        logger.info("🚀 Starting Advanced Setup for Autonomous Trading System")
        logger.info("=" * 80)
        
        setup_steps = [
            ("System Requirements Check", self.check_system_requirements),
            ("Install Ollama", self.install_ollama),
            ("Setup Qwen 2.5 7B Model", self.setup_qwen_model),
            ("Install LangChain Dependencies", self.install_langchain_dependencies),
            ("Create Docker Compose", self.create_docker_compose),
            ("Create Database Schema", self.create_database_schema),
            ("Create Test Scripts", self.create_test_scripts)
        ]
        
        completed_steps = 0
        
        for step_name, step_func in setup_steps:
            logger.info(f"\n{'='*20} {step_name} {'='*20}")
            
            try:
                if step_func():
                    completed_steps += 1
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.error(f"❌ {step_name} failed")
                    break
            except Exception as e:
                logger.error(f"❌ {step_name} failed with exception: {e}")
                break
        
        # Final status report
        logger.info("\n" + "=" * 80)
        logger.info("📊 SETUP SUMMARY")
        logger.info("=" * 80)
        
        for key, status in self.status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"{status_icon} {key.replace('_', ' ').title()}: {'Complete' if status else 'Failed'}")
        
        success_rate = completed_steps / len(setup_steps)
        
        if success_rate == 1.0:
            logger.info("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
            logger.info("Your autonomous trading system is ready for use.")
            
            logger.info("\n📝 Next Steps:")
            logger.info("1. Start services: docker-compose up -d")
            logger.info("2. Test setup: python3 test_ollama.py")
            logger.info("3. Verify database: python3 test_database.py")
            logger.info("4. Test LangChain: python3 test_langchain.py")
            logger.info("5. Run backtesting: python3 src/backtester.py")
            
            return True
        else:
            logger.error(f"\n⚠️ SETUP PARTIALLY COMPLETED ({success_rate:.0%})")
            logger.error("Some components failed to install. Check the logs above.")
            
            logger.info("\n🔧 Troubleshooting:")
            logger.info("1. Check system requirements (RAM, disk space)")
            logger.info("2. Ensure internet connection for downloads")
            logger.info("3. Check permissions for installation")
            logger.info("4. Review error messages in setup_advanced.log")
            
            return False


def main():
    """Main setup function."""
    try:
        setup = AdvancedSetup()
        success = setup.run_full_setup()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n🛑 Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Setup failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()