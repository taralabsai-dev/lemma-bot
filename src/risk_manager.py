#!/usr/bin/env python3
"""
Risk Management Module
Implements comprehensive risk controls and monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Suppress correlation warnings for small datasets
warnings.filterwarnings('ignore', category=RuntimeWarning)

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk severity levels."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskAction(Enum):
    """Risk management actions."""
    MONITOR = "MONITOR"
    REDUCE_POSITION = "REDUCE_POSITION"
    STOP_LOSS = "STOP_LOSS"
    BLOCK_TRADE = "BLOCK_TRADE"
    RISK_OFF_MODE = "RISK_OFF_MODE"


class MarketRegime(Enum):
    """Market volatility regimes."""
    LOW_VOL = "LOW_VOLATILITY"
    NORMAL = "NORMAL"
    HIGH_VOL = "HIGH_VOLATILITY"
    EXTREME_VOL = "EXTREME_VOLATILITY"


@dataclass
class RiskAlert:
    """Risk alert or violation."""
    alert_id: str
    timestamp: str
    risk_type: str
    risk_level: RiskLevel
    ticker: Optional[str]
    current_value: float
    limit_value: float
    message: str
    recommended_action: RiskAction
    acknowledged: bool = False


@dataclass
class StopLossOrder:
    """Stop-loss order tracking."""
    ticker: str
    peak_price: float
    stop_price: float
    current_price: float
    trailing_pct: float
    created_at: str
    last_updated: str
    triggered: bool = False


@dataclass
class RiskMetrics:
    """Portfolio risk metrics snapshot."""
    timestamp: str
    portfolio_value: float
    max_drawdown: float
    portfolio_volatility: float
    var_95: float  # Value at Risk 95%
    market_regime: MarketRegime
    risk_score: float
    sector_concentration: Dict[str, float]
    correlation_risks: List[Dict[str, Any]]


class RiskManager:
    """Comprehensive risk management system."""
    
    def __init__(self, 
                 data_dir: str = "data/risk",
                 max_drawdown_limit: float = 0.20,
                 correlation_limit: float = 0.80,
                 sector_concentration_limit: float = 0.40,
                 trailing_stop_pct: float = 0.15,
                 volatility_lookback: int = 20):
        """Initialize risk manager.
        
        Args:
            data_dir: Directory for risk data storage
            max_drawdown_limit: Maximum portfolio drawdown (20%)
            correlation_limit: Maximum correlation between positions (80%)
            sector_concentration_limit: Maximum sector concentration (40%)
            trailing_stop_pct: Trailing stop percentage (15%)
            volatility_lookback: Days for volatility calculation
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Risk limits
        self.max_drawdown_limit = max_drawdown_limit
        self.correlation_limit = correlation_limit
        self.sector_concentration_limit = sector_concentration_limit
        self.trailing_stop_pct = trailing_stop_pct
        self.volatility_lookback = volatility_lookback
        
        # Risk state
        self.alerts: List[RiskAlert] = []
        self.stop_loss_orders: Dict[str, StopLossOrder] = {}
        self.risk_metrics_history: List[RiskMetrics] = []
        self.current_market_regime = MarketRegime.NORMAL
        self.risk_off_mode = False
        
        # Sector mapping
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
        
        # Load existing risk state
        self._load_risk_state()
    
    def _generate_alert_id(self) -> str:
        """Generate unique alert ID."""
        return f"RISK_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.alerts):03d}"
    
    def update_stop_losses(self, positions: Dict[str, Any], price_data: Dict[str, float]) -> List[str]:
        """Update trailing stop-loss orders.
        
        Args:
            positions: Current portfolio positions
            price_data: Current price data
            
        Returns:
            List of tickers triggered for stop-loss
        """
        triggered_stops = []
        
        for ticker, position in positions.items():
            if ticker not in price_data:
                continue
                
            current_price = price_data[ticker]
            
            # Initialize or update stop-loss order
            if ticker not in self.stop_loss_orders:
                # Create new stop-loss order
                self.stop_loss_orders[ticker] = StopLossOrder(
                    ticker=ticker,
                    peak_price=current_price,
                    stop_price=current_price * (1 - self.trailing_stop_pct),
                    current_price=current_price,
                    trailing_pct=self.trailing_stop_pct,
                    created_at=datetime.now().isoformat(),
                    last_updated=datetime.now().isoformat()
                )
            else:
                stop_order = self.stop_loss_orders[ticker]
                
                # Update peak price if current price is higher
                if current_price > stop_order.peak_price:
                    stop_order.peak_price = current_price
                    stop_order.stop_price = current_price * (1 - self.trailing_stop_pct)
                
                stop_order.current_price = current_price
                stop_order.last_updated = datetime.now().isoformat()
                
                # Check if stop-loss is triggered
                if current_price <= stop_order.stop_price and not stop_order.triggered:
                    stop_order.triggered = True
                    triggered_stops.append(ticker)
                    
                    # Generate risk alert
                    alert = RiskAlert(
                        alert_id=self._generate_alert_id(),
                        timestamp=datetime.now().isoformat(),
                        risk_type="STOP_LOSS",
                        risk_level=RiskLevel.HIGH,
                        ticker=ticker,
                        current_value=current_price,
                        limit_value=stop_order.stop_price,
                        message=f"Trailing stop-loss triggered for {ticker}: "
                               f"${current_price:.2f} <= ${stop_order.stop_price:.2f} "
                               f"({self.trailing_stop_pct:.1%} from peak ${stop_order.peak_price:.2f})",
                        recommended_action=RiskAction.STOP_LOSS
                    )
                    
                    self.alerts.append(alert)
                    logger.warning(f"Stop-loss triggered for {ticker}")
        
        return triggered_stops
    
    def calculate_portfolio_volatility(self, 
                                     returns_data: Dict[str, pd.Series],
                                     weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility.
        
        Args:
            returns_data: Historical returns for each position
            weights: Position weights in portfolio
            
        Returns:
            Portfolio volatility (annualized)
        """
        if not returns_data or not weights:
            return 0.0
        
        # Align returns data
        tickers = list(set(returns_data.keys()) & set(weights.keys()))
        if len(tickers) < 2:
            return 0.0
        
        # Create returns matrix
        returns_df = pd.DataFrame({ticker: returns_data[ticker] for ticker in tickers})
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 5:  # Need minimum data
            return 0.0
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        # Create weight vector
        weight_vector = np.array([weights.get(ticker, 0) for ticker in tickers])
        
        # Portfolio variance
        portfolio_variance = np.dot(weight_vector.T, np.dot(cov_matrix.values, weight_vector))
        
        return np.sqrt(portfolio_variance)
    
    def calculate_correlation_matrix(self, 
                                   returns_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate correlation matrix for positions.
        
        Args:
            returns_data: Historical returns for each position
            
        Returns:
            Correlation matrix DataFrame
        """
        if len(returns_data) < 2:
            return pd.DataFrame()
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 5:
            return pd.DataFrame()
        
        return returns_df.corr()
    
    def check_correlation_risks(self, 
                              returns_data: Dict[str, pd.Series],
                              weights: Dict[str, float]) -> List[Dict[str, Any]]:
        """Check for high correlation risks between positions.
        
        Args:
            returns_data: Historical returns data
            weights: Position weights
            
        Returns:
            List of correlation risk violations
        """
        correlation_risks = []
        
        corr_matrix = self.calculate_correlation_matrix(returns_data)
        if corr_matrix.empty:
            return correlation_risks
        
        # Check pairwise correlations
        for i, ticker1 in enumerate(corr_matrix.columns):
            for j, ticker2 in enumerate(corr_matrix.columns):
                if i >= j:  # Avoid duplicates and self-correlation
                    continue
                
                correlation = corr_matrix.loc[ticker1, ticker2]
                
                if abs(correlation) > self.correlation_limit:
                    # Calculate combined weight
                    combined_weight = weights.get(ticker1, 0) + weights.get(ticker2, 0)
                    
                    risk = {
                        'ticker1': ticker1,
                        'ticker2': ticker2,
                        'correlation': correlation,
                        'combined_weight': combined_weight,
                        'risk_level': RiskLevel.HIGH if abs(correlation) > 0.9 else RiskLevel.MEDIUM
                    }
                    
                    correlation_risks.append(risk)
                    
                    # Generate alert
                    alert = RiskAlert(
                        alert_id=self._generate_alert_id(),
                        timestamp=datetime.now().isoformat(),
                        risk_type="HIGH_CORRELATION",
                        risk_level=risk['risk_level'],
                        ticker=f"{ticker1}-{ticker2}",
                        current_value=abs(correlation),
                        limit_value=self.correlation_limit,
                        message=f"High correlation ({correlation:.2f}) between {ticker1} and {ticker2} "
                               f"(combined weight: {combined_weight:.1%})",
                        recommended_action=RiskAction.REDUCE_POSITION
                    )
                    
                    self.alerts.append(alert)
        
        return correlation_risks
    
    def check_sector_concentration(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """Check sector concentration limits.
        
        Args:
            positions: Current portfolio positions
            
        Returns:
            Dictionary of sector concentrations
        """
        sector_weights = {}
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value <= 0:
            return sector_weights
        
        # Calculate sector weights
        for ticker, position in positions.items():
            sector = self.sector_map.get(ticker, 'Other')
            weight = position.get('market_value', 0) / total_value
            
            if sector not in sector_weights:
                sector_weights[sector] = 0
            sector_weights[sector] += weight
        
        # Check for violations
        for sector, weight in sector_weights.items():
            if weight > self.sector_concentration_limit:
                alert = RiskAlert(
                    alert_id=self._generate_alert_id(),
                    timestamp=datetime.now().isoformat(),
                    risk_type="SECTOR_CONCENTRATION",
                    risk_level=RiskLevel.HIGH if weight > 0.5 else RiskLevel.MEDIUM,
                    ticker=sector,
                    current_value=weight,
                    limit_value=self.sector_concentration_limit,
                    message=f"Sector concentration violation: {sector} = {weight:.1%} "
                           f"(limit: {self.sector_concentration_limit:.1%})",
                    recommended_action=RiskAction.REDUCE_POSITION
                )
                
                self.alerts.append(alert)
        
        return sector_weights
    
    def calculate_value_at_risk(self, 
                              returns_data: Dict[str, pd.Series],
                              weights: Dict[str, float],
                              portfolio_value: float,
                              confidence: float = 0.95) -> float:
        """Calculate Value at Risk (VaR).
        
        Args:
            returns_data: Historical returns
            weights: Position weights
            portfolio_value: Current portfolio value
            confidence: Confidence level (default 95%)
            
        Returns:
            VaR amount in dollars
        """
        if not returns_data or not weights:
            return 0.0
        
        # Calculate portfolio returns
        tickers = list(set(returns_data.keys()) & set(weights.keys()))
        if not tickers:
            return 0.0
        
        portfolio_returns = []
        
        # Get aligned data
        returns_df = pd.DataFrame({ticker: returns_data[ticker] for ticker in tickers})
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 10:
            return 0.0
        
        # Calculate weighted portfolio returns
        for _, row in returns_df.iterrows():
            portfolio_return = sum(row[ticker] * weights.get(ticker, 0) for ticker in tickers)
            portfolio_returns.append(portfolio_return)
        
        # Calculate VaR
        var_percentile = np.percentile(portfolio_returns, (1 - confidence) * 100)
        var_amount = abs(var_percentile * portfolio_value)
        
        return var_amount
    
    def determine_market_regime(self, market_volatility: float) -> MarketRegime:
        """Determine current market volatility regime.
        
        Args:
            market_volatility: Current market volatility
            
        Returns:
            Market regime classification
        """
        if market_volatility < 0.15:
            return MarketRegime.LOW_VOL
        elif market_volatility < 0.25:
            return MarketRegime.NORMAL
        elif market_volatility < 0.40:
            return MarketRegime.HIGH_VOL
        else:
            return MarketRegime.EXTREME_VOL
    
    def calculate_risk_adjusted_position_size(self, 
                                            base_size: float,
                                            stock_volatility: float,
                                            market_regime: MarketRegime) -> float:
        """Adjust position size based on risk factors.
        
        Args:
            base_size: Base position size
            stock_volatility: Individual stock volatility
            market_regime: Current market regime
            
        Returns:
            Risk-adjusted position size
        """
        # Volatility adjustment
        vol_adjustment = max(0.5, min(1.5, 1.0 - (stock_volatility - 0.20) / 0.30))
        
        # Market regime adjustment
        regime_adjustments = {
            MarketRegime.LOW_VOL: 1.1,
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOL: 0.8,
            MarketRegime.EXTREME_VOL: 0.6
        }
        
        regime_adjustment = regime_adjustments.get(market_regime, 1.0)
        
        # Risk-off mode adjustment
        risk_off_adjustment = 0.5 if self.risk_off_mode else 1.0
        
        adjusted_size = base_size * vol_adjustment * regime_adjustment * risk_off_adjustment
        
        return max(0, adjusted_size)
    
    def evaluate_trade_risk(self, 
                          ticker: str,
                          order_type: str,
                          size: float,
                          current_positions: Dict[str, Any],
                          price_data: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Evaluate if a trade should be allowed based on risk limits.
        
        Args:
            ticker: Stock ticker
            order_type: 'BUY' or 'SELL'
            size: Trade size (shares or weight)
            current_positions: Current portfolio positions
            price_data: Current price data
            
        Returns:
            Tuple of (trade_allowed, risk_reasons)
        """
        risk_reasons = []
        
        # Check if in risk-off mode
        if self.risk_off_mode and order_type == 'BUY':
            risk_reasons.append("Risk-off mode active: blocking new buy orders")
        
        # Check stop-loss violations
        if ticker in self.stop_loss_orders and self.stop_loss_orders[ticker].triggered:
            risk_reasons.append(f"Stop-loss triggered for {ticker}")
        
        # For buy orders, check additional constraints
        if order_type == 'BUY':
            # Check sector concentration
            sector = self.sector_map.get(ticker, 'Other')
            sector_weights = self.check_sector_concentration(current_positions)
            current_sector_weight = sector_weights.get(sector, 0)
            
            # Estimate new sector weight (rough calculation)
            total_value = sum(pos.get('market_value', 0) for pos in current_positions.values())
            if total_value > 0 and ticker in price_data:
                additional_value = size * price_data[ticker]
                new_sector_weight = (current_sector_weight * total_value + additional_value) / (total_value + additional_value)
                
                if new_sector_weight > self.sector_concentration_limit:
                    risk_reasons.append(f"Would exceed sector concentration limit for {sector}: "
                                      f"{new_sector_weight:.1%} > {self.sector_concentration_limit:.1%}")
        
        # Check extreme volatility regime
        if self.current_market_regime == MarketRegime.EXTREME_VOL and order_type == 'BUY':
            risk_reasons.append("Extreme volatility detected: limiting new positions")
        
        trade_allowed = len(risk_reasons) == 0
        
        if not trade_allowed:
            # Generate risk alert
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                timestamp=datetime.now().isoformat(),
                risk_type="TRADE_BLOCK",
                risk_level=RiskLevel.HIGH,
                ticker=ticker,
                current_value=size,
                limit_value=0,
                message=f"Trade blocked for {ticker}: {'; '.join(risk_reasons)}",
                recommended_action=RiskAction.BLOCK_TRADE
            )
            
            self.alerts.append(alert)
        
        return trade_allowed, risk_reasons
    
    def update_risk_metrics(self, 
                          portfolio_value: float,
                          positions: Dict[str, Any],
                          returns_data: Dict[str, pd.Series],
                          max_drawdown: float) -> RiskMetrics:
        """Update comprehensive risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            returns_data: Historical returns data
            max_drawdown: Current maximum drawdown
            
        Returns:
            RiskMetrics object
        """
        # Calculate position weights
        weights = {}
        if portfolio_value > 0:
            for ticker, position in positions.items():
                weights[ticker] = position.get('market_value', 0) / portfolio_value
        
        # Portfolio volatility
        portfolio_vol = self.calculate_portfolio_volatility(returns_data, weights)
        
        # Market regime
        self.current_market_regime = self.determine_market_regime(portfolio_vol)
        
        # Value at Risk
        var_95 = self.calculate_value_at_risk(returns_data, weights, portfolio_value)
        
        # Sector concentration
        sector_concentration = self.check_sector_concentration(positions)
        
        # Correlation risks
        correlation_risks = self.check_correlation_risks(returns_data, weights)
        
        # Risk score (0-100, higher is riskier)
        risk_score = self._calculate_risk_score(
            max_drawdown, portfolio_vol, sector_concentration, correlation_risks
        )
        
        # Check for risk-off mode trigger
        self._check_risk_off_mode(max_drawdown, portfolio_vol, risk_score)
        
        # Create risk metrics snapshot
        risk_metrics = RiskMetrics(
            timestamp=datetime.now().isoformat(),
            portfolio_value=portfolio_value,
            max_drawdown=max_drawdown,
            portfolio_volatility=portfolio_vol,
            var_95=var_95,
            market_regime=self.current_market_regime,
            risk_score=risk_score,
            sector_concentration=sector_concentration,
            correlation_risks=correlation_risks
        )
        
        self.risk_metrics_history.append(risk_metrics)
        
        # Keep only last 100 snapshots
        if len(self.risk_metrics_history) > 100:
            self.risk_metrics_history = self.risk_metrics_history[-100:]
        
        return risk_metrics
    
    def _calculate_risk_score(self, 
                            max_drawdown: float,
                            portfolio_vol: float,
                            sector_concentration: Dict[str, float],
                            correlation_risks: List[Dict]) -> float:
        """Calculate overall portfolio risk score.
        
        Args:
            max_drawdown: Current maximum drawdown
            portfolio_vol: Portfolio volatility
            sector_concentration: Sector weights
            correlation_risks: High correlation risks
            
        Returns:
            Risk score (0-100)
        """
        score = 0
        
        # Drawdown component (0-30 points)
        drawdown_score = min(30, (max_drawdown / self.max_drawdown_limit) * 30)
        score += drawdown_score
        
        # Volatility component (0-25 points)
        vol_score = min(25, (portfolio_vol / 0.30) * 25)  # 30% vol = max score
        score += vol_score
        
        # Sector concentration component (0-25 points)
        max_sector_weight = max(sector_concentration.values()) if sector_concentration else 0
        concentration_score = min(25, (max_sector_weight / self.sector_concentration_limit) * 25)
        score += concentration_score
        
        # Correlation component (0-20 points)
        correlation_score = min(20, len(correlation_risks) * 5)
        score += correlation_score
        
        return min(100, score)
    
    def _check_risk_off_mode(self, max_drawdown: float, portfolio_vol: float, risk_score: float) -> None:
        """Check if risk-off mode should be activated.
        
        Args:
            max_drawdown: Current maximum drawdown
            portfolio_vol: Portfolio volatility
            risk_score: Overall risk score
        """
        # Trigger conditions for risk-off mode
        trigger_conditions = []
        
        if max_drawdown > self.max_drawdown_limit * 0.8:  # 80% of max drawdown limit
            trigger_conditions.append(f"Drawdown approaching limit: {max_drawdown:.1%}")
        
        if portfolio_vol > 0.35:  # 35% volatility threshold
            trigger_conditions.append(f"High portfolio volatility: {portfolio_vol:.1%}")
        
        if risk_score > 75:  # High risk score
            trigger_conditions.append(f"High risk score: {risk_score:.0f}")
        
        if self.current_market_regime == MarketRegime.EXTREME_VOL:
            trigger_conditions.append("Extreme market volatility regime")
        
        # Activate risk-off mode
        if trigger_conditions and not self.risk_off_mode:
            self.risk_off_mode = True
            
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                timestamp=datetime.now().isoformat(),
                risk_type="RISK_OFF_MODE",
                risk_level=RiskLevel.CRITICAL,
                ticker=None,
                current_value=risk_score,
                limit_value=75,
                message=f"Risk-off mode activated: {'; '.join(trigger_conditions)}",
                recommended_action=RiskAction.RISK_OFF_MODE
            )
            
            self.alerts.append(alert)
            logger.critical("Risk-off mode activated")
        
        # Deactivate risk-off mode
        elif not trigger_conditions and self.risk_off_mode:
            self.risk_off_mode = False
            
            alert = RiskAlert(
                alert_id=self._generate_alert_id(),
                timestamp=datetime.now().isoformat(),
                risk_type="RISK_OFF_MODE",
                risk_level=RiskLevel.LOW,
                ticker=None,
                current_value=risk_score,
                limit_value=75,
                message="Risk-off mode deactivated - risk conditions normalized",
                recommended_action=RiskAction.MONITOR
            )
            
            self.alerts.append(alert)
            logger.info("Risk-off mode deactivated")
    
    def get_active_alerts(self, risk_level: Optional[RiskLevel] = None) -> List[RiskAlert]:
        """Get active risk alerts.
        
        Args:
            risk_level: Filter by risk level
            
        Returns:
            List of active alerts
        """
        active_alerts = [alert for alert in self.alerts if not alert.acknowledged]
        
        if risk_level:
            active_alerts = [alert for alert in active_alerts if alert.risk_level == risk_level]
        
        # Sort by risk level and timestamp
        risk_priority = {
            RiskLevel.CRITICAL: 4,
            RiskLevel.HIGH: 3,
            RiskLevel.MEDIUM: 2,
            RiskLevel.LOW: 1
        }
        
        active_alerts.sort(key=lambda x: (risk_priority.get(x.risk_level, 0), x.timestamp), reverse=True)
        
        return active_alerts
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a risk alert.
        
        Args:
            alert_id: Alert ID to acknowledge
            
        Returns:
            True if alert was found and acknowledged
        """
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                logger.info(f"Risk alert acknowledged: {alert_id}")
                return True
        
        return False
    
    def generate_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report.
        
        Returns:
            Dictionary with risk report data
        """
        latest_metrics = self.risk_metrics_history[-1] if self.risk_metrics_history else None
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'risk_off_mode': self.risk_off_mode,
            'market_regime': self.current_market_regime.value if latest_metrics else 'UNKNOWN',
            'risk_score': latest_metrics.risk_score if latest_metrics else 0,
            'active_alerts': len(self.get_active_alerts()),
            'critical_alerts': len(self.get_active_alerts(RiskLevel.CRITICAL)),
            'high_alerts': len(self.get_active_alerts(RiskLevel.HIGH)),
            'stop_loss_orders': len([order for order in self.stop_loss_orders.values() if not order.triggered]),
            'triggered_stops': len([order for order in self.stop_loss_orders.values() if order.triggered]),
            'latest_metrics': asdict(latest_metrics) if latest_metrics else None
        }
        
        return report
    
    def _save_risk_state(self) -> None:
        """Save risk manager state to disk."""
        try:
            state = {
                'risk_off_mode': self.risk_off_mode,
                'current_market_regime': self.current_market_regime.value,
                'alerts': [asdict(alert) for alert in self.alerts[-50:]],  # Keep last 50 alerts
                'stop_loss_orders': {k: asdict(v) for k, v in self.stop_loss_orders.items()},
                'risk_metrics_history': [asdict(m) for m in self.risk_metrics_history[-20:]]  # Keep last 20
            }
            
            # Convert enums to strings
            for alert in state['alerts']:
                if 'risk_level' in alert:
                    alert['risk_level'] = alert['risk_level'].value if hasattr(alert['risk_level'], 'value') else str(alert['risk_level'])
                if 'recommended_action' in alert:
                    alert['recommended_action'] = alert['recommended_action'].value if hasattr(alert['recommended_action'], 'value') else str(alert['recommended_action'])
            
            for metrics in state['risk_metrics_history']:
                if 'market_regime' in metrics:
                    metrics['market_regime'] = metrics['market_regime'].value if hasattr(metrics['market_regime'], 'value') else str(metrics['market_regime'])
            
            state_file = self.data_dir / "risk_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving risk state: {e}")
    
    def _load_risk_state(self) -> None:
        """Load risk manager state from disk."""
        try:
            state_file = self.data_dir / "risk_state.json"
            if not state_file.exists():
                return
            
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            self.risk_off_mode = state.get('risk_off_mode', False)
            self.current_market_regime = MarketRegime(state.get('current_market_regime', 'NORMAL'))
            
            # Load alerts
            alerts_data = state.get('alerts', [])
            for alert_data in alerts_data:
                alert_data['risk_level'] = RiskLevel(alert_data['risk_level'])
                alert_data['recommended_action'] = RiskAction(alert_data['recommended_action'])
                self.alerts.append(RiskAlert(**alert_data))
            
            # Load stop-loss orders
            orders_data = state.get('stop_loss_orders', {})
            for ticker, order_data in orders_data.items():
                self.stop_loss_orders[ticker] = StopLossOrder(**order_data)
            
            # Load risk metrics
            metrics_data = state.get('risk_metrics_history', [])
            for metrics in metrics_data:
                metrics['market_regime'] = MarketRegime(metrics['market_regime'])
                self.risk_metrics_history.append(RiskMetrics(**metrics))
            
            logger.info("Risk manager state loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading risk state: {e}")


def main():
    """Main function for standalone testing."""
    import argparse
    import sys
    import os
    
    # Add parent directory for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from portfolio_manager import PortfolioManager
    from data_collector import MarketDataCollector
    
    parser = argparse.ArgumentParser(description='Risk Manager')
    parser.add_argument('--status', action='store_true', help='Show risk status')
    parser.add_argument('--alerts', action='store_true', help='Show active alerts')
    parser.add_argument('--report', action='store_true', help='Generate risk report')
    parser.add_argument('--test-stops', action='store_true', help='Test stop-loss functionality')
    
    args = parser.parse_args()
    
    # Initialize risk manager
    risk_manager = RiskManager()
    
    if args.status:
        # Show risk status
        print("\n" + "="*80)
        print("RISK MANAGEMENT STATUS")
        print("="*80)
        
        print(f"\nRisk-Off Mode: {'ACTIVE' if risk_manager.risk_off_mode else 'INACTIVE'}")
        print(f"Market Regime: {risk_manager.current_market_regime.value}")
        
        active_alerts = risk_manager.get_active_alerts()
        print(f"Active Alerts: {len(active_alerts)}")
        
        if risk_manager.risk_metrics_history:
            latest = risk_manager.risk_metrics_history[-1]
            print(f"Risk Score: {latest.risk_score:.0f}/100")
            print(f"Portfolio Volatility: {latest.portfolio_volatility:.1%}")
            print(f"Max Drawdown: {latest.max_drawdown:.1%}")
            print(f"VaR (95%): ${latest.var_95:,.0f}")
        
        print(f"\nStop-Loss Orders:")
        active_stops = [order for order in risk_manager.stop_loss_orders.values() if not order.triggered]
        triggered_stops = [order for order in risk_manager.stop_loss_orders.values() if order.triggered]
        print(f"  Active: {len(active_stops)}")
        print(f"  Triggered: {len(triggered_stops)}")
    
    elif args.alerts:
        # Show active alerts
        active_alerts = risk_manager.get_active_alerts()
        
        print("\n" + "="*80)
        print("ACTIVE RISK ALERTS")
        print("="*80)
        
        if not active_alerts:
            print("\nNo active alerts.")
        else:
            for alert in active_alerts:
                print(f"\n[{alert.risk_level.value}] {alert.risk_type}")
                print(f"  Time: {alert.timestamp[:19]}")
                if alert.ticker:
                    print(f"  Ticker: {alert.ticker}")
                print(f"  Current: {alert.current_value:.3f} | Limit: {alert.limit_value:.3f}")
                print(f"  Message: {alert.message}")
                print(f"  Action: {alert.recommended_action.value}")
    
    elif args.report:
        # Generate risk report
        report = risk_manager.generate_risk_report()
        
        print("\n" + "="*80)
        print("COMPREHENSIVE RISK REPORT")
        print("="*80)
        
        print(f"\nOverall Status:")
        print(f"  Risk-Off Mode: {report['risk_off_mode']}")
        print(f"  Market Regime: {report['market_regime']}")
        print(f"  Risk Score: {report['risk_score']:.0f}/100")
        
        print(f"\nAlert Summary:")
        print(f"  Total Active: {report['active_alerts']}")
        print(f"  Critical: {report['critical_alerts']}")
        print(f"  High: {report['high_alerts']}")
        
        print(f"\nStop-Loss Summary:")
        print(f"  Active Orders: {report['stop_loss_orders']}")
        print(f"  Triggered: {report['triggered_stops']}")
        
        if report['latest_metrics']:
            metrics = report['latest_metrics']
            print(f"\nRisk Metrics:")
            print(f"  Portfolio Value: ${metrics['portfolio_value']:,.0f}")
            print(f"  Max Drawdown: {metrics['max_drawdown']:.1%}")
            print(f"  Volatility: {metrics['portfolio_volatility']:.1%}")
            print(f"  VaR (95%): ${metrics['var_95']:,.0f}")
            
            if metrics['sector_concentration']:
                print(f"\nSector Concentration:")
                for sector, weight in metrics['sector_concentration'].items():
                    print(f"  {sector}: {weight:.1%}")
    
    elif args.test_stops:
        # Test stop-loss functionality with dummy data
        print("Testing stop-loss functionality...")
        
        # Dummy positions and prices
        positions = {
            'AAPL': {'market_value': 1000},
            'MSFT': {'market_value': 1500},
            'NVDA': {'market_value': 2000}
        }
        
        # Initial prices
        initial_prices = {'AAPL': 200, 'MSFT': 300, 'NVDA': 400}
        
        # Update with initial prices
        triggered = risk_manager.update_stop_losses(positions, initial_prices)
        print(f"Initial stop-losses set. Triggered: {triggered}")
        
        # Simulate price increases
        higher_prices = {'AAPL': 220, 'MSFT': 330, 'NVDA': 450}
        triggered = risk_manager.update_stop_losses(positions, higher_prices)
        print(f"Prices increased. Triggered: {triggered}")
        
        # Simulate price drop triggering stop-loss
        drop_prices = {'AAPL': 180, 'MSFT': 270, 'NVDA': 350}  # AAPL drops below 15% stop
        triggered = risk_manager.update_stop_losses(positions, drop_prices)
        print(f"Prices dropped. Triggered: {triggered}")
        
        # Show stop-loss orders
        for ticker, order in risk_manager.stop_loss_orders.items():
            print(f"\n{ticker}:")
            print(f"  Peak: ${order.peak_price:.2f}")
            print(f"  Stop: ${order.stop_price:.2f}")
            print(f"  Current: ${order.current_price:.2f}")
            print(f"  Triggered: {order.triggered}")
    
    else:
        print("Use --status, --alerts, --report, or --test-stops")


if __name__ == "__main__":
    main()