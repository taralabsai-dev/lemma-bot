#!/usr/bin/env python3
"""
Trade Approval System
Manages pending trades and approval workflow
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import os

# PDF generation
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib import colors
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("reportlab not available - PDF export disabled")

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Status of pending trades."""
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXECUTED = "EXECUTED"
    MODIFIED = "MODIFIED"


class TradeAction(Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class PendingTrade:
    """Represents a pending trade awaiting approval."""
    trade_id: str
    ticker: str
    action: TradeAction
    recommended_shares: float
    current_shares: float  # For modification
    current_price: float
    estimated_value: float
    confidence_score: float
    signal_strength: float
    reasoning: str
    llm_analysis: str
    expected_weight_change: float
    risk_impact: str
    sector: str
    status: TradeStatus
    created_at: str
    modified_at: Optional[str] = None
    approved_by: Optional[str] = None
    rejection_reason: Optional[str] = None


class TradeApprovalSystem:
    """Manages trade approval workflow and notifications."""
    
    def __init__(self, data_dir: str = "data/approvals"):
        """Initialize trade approval system."""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.pending_trades: List[PendingTrade] = []
        self.trade_counter = 0
        
        # Email configuration (load from environment)
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', '587')),
            'email_user': os.getenv('EMAIL_USER', ''),
            'email_password': os.getenv('EMAIL_PASSWORD', ''),
            'notification_emails': os.getenv('NOTIFICATION_EMAILS', '').split(',')
        }
        
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
            'PYPL': 'Fintech'
        }
        
        # Load existing pending trades
        self._load_pending_trades()
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        self.trade_counter += 1
        return f"TRADE_{datetime.now().strftime('%Y%m%d')}_{self.trade_counter:04d}"
    
    def create_pending_trade(self,
                           ticker: str,
                           action: TradeAction,
                           shares: float,
                           price: float,
                           reasoning: str,
                           llm_analysis: str,
                           confidence_score: float,
                           signal_strength: float,
                           portfolio_impact: Dict[str, Any]) -> PendingTrade:
        """Create a new pending trade."""
        
        trade = PendingTrade(
            trade_id=self._generate_trade_id(),
            ticker=ticker,
            action=action,
            recommended_shares=shares,
            current_shares=shares,  # Initially same as recommended
            current_price=price,
            estimated_value=shares * price,
            confidence_score=confidence_score,
            signal_strength=signal_strength,
            reasoning=reasoning,
            llm_analysis=llm_analysis,
            expected_weight_change=portfolio_impact.get('weight_change', 0),
            risk_impact=portfolio_impact.get('risk_description', 'Moderate'),
            sector=self.sector_map.get(ticker, 'Other'),
            status=TradeStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        self.pending_trades.append(trade)
        self._save_pending_trades()
        
        logger.info(f"Created pending trade: {trade.trade_id}")
        return trade
    
    def get_pending_trades(self, status: Optional[TradeStatus] = None) -> List[PendingTrade]:
        """Get pending trades, optionally filtered by status."""
        if status is None:
            return [t for t in self.pending_trades if t.status == TradeStatus.PENDING]
        return [t for t in self.pending_trades if t.status == status]
    
    def approve_trade(self, trade_id: str, approved_by: str = "User") -> bool:
        """Approve a pending trade."""
        for trade in self.pending_trades:
            if trade.trade_id == trade_id and trade.status == TradeStatus.PENDING:
                trade.status = TradeStatus.APPROVED
                trade.approved_by = approved_by
                trade.modified_at = datetime.now().isoformat()
                
                self._save_pending_trades()
                logger.info(f"Trade approved: {trade_id}")
                return True
        
        return False
    
    def reject_trade(self, trade_id: str, reason: str = "", rejected_by: str = "User") -> bool:
        """Reject a pending trade."""
        for trade in self.pending_trades:
            if trade.trade_id == trade_id and trade.status == TradeStatus.PENDING:
                trade.status = TradeStatus.REJECTED
                trade.rejection_reason = reason
                trade.approved_by = rejected_by  # Who rejected it
                trade.modified_at = datetime.now().isoformat()
                
                self._save_pending_trades()
                logger.info(f"Trade rejected: {trade_id}")
                return True
        
        return False
    
    def modify_trade_size(self, trade_id: str, new_shares: float) -> bool:
        """Modify trade size before approval."""
        for trade in self.pending_trades:
            if trade.trade_id == trade_id and trade.status == TradeStatus.PENDING:
                trade.current_shares = new_shares
                trade.estimated_value = new_shares * trade.current_price
                trade.status = TradeStatus.MODIFIED
                trade.modified_at = datetime.now().isoformat()
                
                self._save_pending_trades()
                logger.info(f"Trade size modified: {trade_id} -> {new_shares} shares")
                return True
        
        return False
    
    def bulk_approve_trades(self, trade_ids: List[str], approved_by: str = "User") -> int:
        """Bulk approve multiple trades."""
        approved_count = 0
        
        for trade_id in trade_ids:
            if self.approve_trade(trade_id, approved_by):
                approved_count += 1
        
        logger.info(f"Bulk approved {approved_count} trades")
        return approved_count
    
    def get_portfolio_impact_summary(self) -> Dict[str, Any]:
        """Calculate summary of portfolio impact from pending trades."""
        pending = self.get_pending_trades(TradeStatus.PENDING)
        
        if not pending:
            return {
                'total_trades': 0,
                'buy_count': 0,
                'sell_count': 0,
                'net_capital_change': 0,
                'sectors_affected': [],
                'high_confidence_trades': 0
            }
        
        buy_trades = [t for t in pending if t.action == TradeAction.BUY]
        sell_trades = [t for t in pending if t.action == TradeAction.SELL]
        
        buy_value = sum(t.estimated_value for t in buy_trades)
        sell_value = sum(t.estimated_value for t in sell_trades)
        
        sectors_affected = list(set(t.sector for t in pending))
        high_confidence = len([t for t in pending if t.confidence_score > 0.7])
        
        return {
            'total_trades': len(pending),
            'buy_count': len(buy_trades),
            'sell_count': len(sell_trades),
            'net_capital_change': sell_value - buy_value,
            'sectors_affected': sectors_affected,
            'high_confidence_trades': high_confidence,
            'avg_confidence': np.mean([t.confidence_score for t in pending]),
            'avg_signal_strength': np.mean([t.signal_strength for t in pending])
        }
    
    def generate_weekly_summary_pdf(self, output_path: Optional[str] = None) -> str:
        """Generate PDF summary of weekly trading decisions."""
        if not PDF_AVAILABLE:
            raise ImportError("reportlab package required for PDF generation")
        
        if output_path is None:
            output_path = self.data_dir / f"weekly_summary_{datetime.now().strftime('%Y%m%d')}.pdf"
        
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            textColor=colors.HexColor('#1f77b4')
        )
        story.append(Paragraph("Autonomous Trading System - Weekly Decision Summary", title_style))
        story.append(Spacer(1, 12))
        
        # Date range
        week_start = datetime.now() - timedelta(days=7)
        story.append(Paragraph(f"Period: {week_start.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Portfolio Impact Summary
        impact = self.get_portfolio_impact_summary()
        story.append(Paragraph("Portfolio Impact Summary", styles['Heading2']))
        
        impact_data = [
            ['Metric', 'Value'],
            ['Total Pending Trades', str(impact['total_trades'])],
            ['Buy Orders', str(impact['buy_count'])],
            ['Sell Orders', str(impact['sell_count'])],
            ['Net Capital Change', f"${impact['net_capital_change']:,.0f}"],
            ['High Confidence Trades', str(impact['high_confidence_trades'])],
            ['Average Confidence', f"{impact.get('avg_confidence', 0):.1%}"],
            ['Average Signal Strength', f"{impact.get('avg_signal_strength', 0):.3f}"],
            ['Sectors Affected', ', '.join(impact['sectors_affected'][:3])]
        ]
        
        impact_table = Table(impact_data, colWidths=[2.5*inch, 2*inch])
        impact_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(impact_table)
        story.append(Spacer(1, 20))
        
        # Pending Trades Detail
        pending_trades = self.get_pending_trades()
        if pending_trades:
            story.append(Paragraph("Pending Trades Detail", styles['Heading2']))
            
            trade_data = [['Ticker', 'Action', 'Shares', 'Value', 'Confidence', 'Reasoning']]
            
            for trade in pending_trades:
                reasoning_short = trade.reasoning[:50] + "..." if len(trade.reasoning) > 50 else trade.reasoning
                trade_data.append([
                    trade.ticker,
                    trade.action.value,
                    f"{trade.current_shares:.0f}",
                    f"${trade.estimated_value:,.0f}",
                    f"{trade.confidence_score:.1%}",
                    reasoning_short
                ])
            
            trade_table = Table(trade_data, colWidths=[0.8*inch, 0.6*inch, 0.8*inch, 1*inch, 0.8*inch, 2*inch])
            trade_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(trade_table)
        
        # Build PDF
        doc.build(story)
        logger.info(f"Generated weekly summary PDF: {output_path}")
        
        return str(output_path)
    
    def send_notification_email(self, subject: str, body: str, attachment_path: Optional[str] = None) -> bool:
        """Send email notification about pending trades."""
        if not self.email_config['email_user'] or not self.email_config['notification_emails']:
            logger.warning("Email configuration incomplete - skipping notification")
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['email_user']
            msg['To'] = ', '.join(self.email_config['notification_emails'])
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachment if provided
            if attachment_path and os.path.exists(attachment_path):
                with open(attachment_path, "rb") as attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment.read())
                
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {os.path.basename(attachment_path)}'
                )
                msg.attach(part)
            
            # Send email
            server = smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port'])
            server.starttls()
            server.login(self.email_config['email_user'], self.email_config['email_password'])
            
            text = msg.as_string()
            server.sendmail(self.email_config['email_user'], self.email_config['notification_emails'], text)
            server.quit()
            
            logger.info("Notification email sent successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send notification email: {e}")
            return False
    
    def notify_pending_trades(self) -> bool:
        """Send notification about new pending trades."""
        pending = self.get_pending_trades()
        
        if not pending:
            return True
        
        # Generate email content
        subject = f"Autonomous Trading: {len(pending)} Trades Awaiting Approval"
        
        impact = self.get_portfolio_impact_summary()
        
        body = f"""
        <html>
        <body>
            <h2>Trading System Alert</h2>
            <p>Your autonomous trading system has generated <strong>{len(pending)} trades</strong> that require approval.</p>
            
            <h3>Summary:</h3>
            <ul>
                <li>Buy Orders: {impact['buy_count']}</li>
                <li>Sell Orders: {impact['sell_count']}</li>
                <li>Net Capital Change: ${impact['net_capital_change']:,.0f}</li>
                <li>High Confidence Trades: {impact['high_confidence_trades']}</li>
                <li>Average Confidence: {impact.get('avg_confidence', 0):.1%}</li>
            </ul>
            
            <h3>Pending Trades:</h3>
            <table border="1" style="border-collapse: collapse;">
                <tr style="background-color: #1f77b4; color: white;">
                    <th>Ticker</th>
                    <th>Action</th>
                    <th>Shares</th>
                    <th>Value</th>
                    <th>Confidence</th>
                </tr>
        """
        
        for trade in pending:
            body += f"""
                <tr>
                    <td>{trade.ticker}</td>
                    <td>{trade.action.value}</td>
                    <td>{trade.current_shares:.0f}</td>
                    <td>${trade.estimated_value:,.0f}</td>
                    <td>{trade.confidence_score:.1%}</td>
                </tr>
            """
        
        body += """
            </table>
            
            <p><strong>Please review and approve these trades in the dashboard.</strong></p>
            <p>Dashboard URL: <a href="http://127.0.0.1:8050">http://127.0.0.1:8050</a></p>
            
            <p>Best regards,<br>Autonomous Trading System</p>
        </body>
        </html>
        """
        
        return self.send_notification_email(subject, body)
    
    def _save_pending_trades(self) -> None:
        """Save pending trades to disk."""
        try:
            trades_data = []
            for trade in self.pending_trades:
                trade_dict = asdict(trade)
                # Convert enums to strings for JSON serialization
                trade_dict['action'] = trade_dict['action'].value if hasattr(trade_dict['action'], 'value') else str(trade_dict['action'])
                trade_dict['status'] = trade_dict['status'].value if hasattr(trade_dict['status'], 'value') else str(trade_dict['status'])
                trades_data.append(trade_dict)
            
            trades_file = self.data_dir / "pending_trades.json"
            with open(trades_file, 'w') as f:
                json.dump({
                    'trades': trades_data,
                    'trade_counter': self.trade_counter
                }, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving pending trades: {e}")
    
    def _load_pending_trades(self) -> None:
        """Load pending trades from disk."""
        try:
            trades_file = self.data_dir / "pending_trades.json"
            if not trades_file.exists():
                return
            
            with open(trades_file, 'r') as f:
                data = json.load(f)
            
            self.trade_counter = data.get('trade_counter', 0)
            
            for trade_data in data.get('trades', []):
                # Convert strings back to enums
                trade_data['action'] = TradeAction(trade_data['action'])
                trade_data['status'] = TradeStatus(trade_data['status'])
                
                trade = PendingTrade(**trade_data)
                self.pending_trades.append(trade)
            
            logger.info(f"Loaded {len(self.pending_trades)} pending trades")
            
        except Exception as e:
            logger.error(f"Error loading pending trades: {e}")


def main():
    """Test the trade approval system."""
    approval_system = TradeApprovalSystem()
    
    # Create some sample pending trades
    print("Creating sample pending trades...")
    
    sample_trades = [
        {
            'ticker': 'AAPL',
            'action': TradeAction.BUY,
            'shares': 100,
            'price': 225.50,
            'reasoning': 'Strong earnings momentum and positive analyst upgrades',
            'llm_analysis': 'Bullish sentiment with 85% confidence based on recent news analysis',
            'confidence_score': 0.85,
            'signal_strength': 0.78,
            'portfolio_impact': {'weight_change': 0.05, 'risk_description': 'Low risk addition'}
        },
        {
            'ticker': 'TSLA',
            'action': TradeAction.SELL,
            'shares': 50,
            'price': 195.40,
            'reasoning': 'Trailing stop-loss triggered at 15% below peak',
            'llm_analysis': 'Risk management sell signal with 95% confidence',
            'confidence_score': 0.95,
            'signal_strength': 0.92,
            'portfolio_impact': {'weight_change': -0.03, 'risk_description': 'Risk reduction'}
        }
    ]
    
    for trade_data in sample_trades:
        trade = approval_system.create_pending_trade(**trade_data)
        print(f"Created trade: {trade.trade_id}")
    
    # Test PDF generation
    if PDF_AVAILABLE:
        print("Generating PDF summary...")
        pdf_path = approval_system.generate_weekly_summary_pdf()
        print(f"PDF generated: {pdf_path}")
    else:
        print("PDF generation not available (reportlab not installed)")
    
    # Test email notification (will skip if not configured)
    print("Sending notification email...")
    approval_system.notify_pending_trades()
    
    print("Trade approval system test completed!")


if __name__ == "__main__":
    main()