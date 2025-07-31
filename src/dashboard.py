#!/usr/bin/env python3
"""
Autonomous Trading System Dashboard
Real-time monitoring dashboard using Dash and Plotly
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import json
import logging
import sys
import os
from typing import Dict, List, Any, Optional

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_manager import PortfolioManager
from risk_manager import RiskManager
from data_collector import MarketDataCollector
from signals.signal_aggregator import SignalAggregator
from llm_analyst import LLMAnalyst
from trade_approval_system import TradeApprovalSystem, PendingTrade, TradeAction, TradeStatus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for components
portfolio_manager = None
risk_manager = None
data_collector = None
signal_aggregator = None
llm_analyst = None
trade_approval_system = None

# Initialize Dash app
app = dash.Dash(__name__, title="Autonomous Trading Dashboard")
app.config.suppress_callback_exceptions = True

# Custom CSS styling
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'
]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# Color scheme
COLORS = {
    'primary': '#1f77b4',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff7f0e',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'background': '#ffffff',
    'text': '#2c3e50'
}

def initialize_components():
    """Initialize all trading system components."""
    global portfolio_manager, risk_manager, data_collector, signal_aggregator, llm_analyst, trade_approval_system
    
    try:
        portfolio_manager = PortfolioManager()
        risk_manager = RiskManager()
        data_collector = MarketDataCollector()
        signal_aggregator = SignalAggregator()
        llm_analyst = LLMAnalyst()
        trade_approval_system = TradeApprovalSystem()
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {e}")

def is_market_open() -> bool:
    """Check if market is currently open (EST)."""
    now = datetime.now()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    current_time = now.time()
    
    # Market is closed on weekends
    if weekday >= 5:  # Saturday or Sunday
        return False
    
    # Market hours: 9:30 AM - 4:00 PM EST
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    return market_open <= current_time <= market_close

def get_benchmark_data() -> pd.DataFrame:
    """Get S&P 500 benchmark data (simplified)."""
    # In a real implementation, you would fetch actual S&P 500 data
    # For demo purposes, generate synthetic benchmark data
    dates = pd.date_range(start='2024-01-01', end='2025-01-31', freq='D')
    
    # Generate synthetic S&P 500 returns (roughly 10% annual with 16% volatility)
    np.random.seed(42)  # For reproducible results
    daily_returns = np.random.normal(0.0003, 0.01, len(dates))  # ~0.03% daily, 1% daily vol
    
    # Create cumulative returns
    cumulative_returns = (1 + pd.Series(daily_returns)).cumprod()
    benchmark_values = 10000 * cumulative_returns  # Start with $10,000
    
    return pd.DataFrame({
        'date': dates,
        'value': benchmark_values,
        'daily_return': daily_returns
    })

def create_header():
    """Create dashboard header."""
    return html.Div([
        html.Div([
            html.H1([
                html.I(className="fas fa-robot", style={'marginRight': '10px'}),
                "Autonomous Trading Dashboard"
            ], style={'color': COLORS['primary'], 'marginBottom': '0'}),
            html.P(
                f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S EST')}",
                style={'color': COLORS['text'], 'fontSize': '14px', 'marginBottom': '0'}
            )
        ], className="eight columns"),
        
        html.Div([
            html.Div(id="market-status", children=[
                html.I(className="fas fa-circle", style={
                    'color': COLORS['success'] if is_market_open() else COLORS['danger'],
                    'marginRight': '5px'
                }),
                html.Span("Market Open" if is_market_open() else "Market Closed")
            ], style={'textAlign': 'right', 'fontSize': '16px', 'fontWeight': 'bold'})
        ], className="four columns")
    ], className="row", style={'marginBottom': '20px'})

def create_summary_cards(portfolio_data: Dict[str, Any]) -> html.Div:
    """Create summary cards for key metrics."""
    cards = []
    
    # Portfolio Value Card
    cards.append(
        html.Div([
            html.Div([
                html.I(className="fas fa-wallet", style={'fontSize': '24px', 'color': COLORS['primary']}),
                html.H3(f"${portfolio_data.get('total_value', 0):,.0f}", style={'margin': '10px 0 5px 0'}),
                html.P("Portfolio Value", style={'margin': '0', 'color': COLORS['text']})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className="three columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'margin': '5px'
        })
    )
    
    # Daily Return Card
    daily_return = portfolio_data.get('daily_return', 0)
    return_color = COLORS['success'] if daily_return >= 0 else COLORS['danger']
    
    cards.append(
        html.Div([
            html.Div([
                html.I(className="fas fa-chart-line", style={'fontSize': '24px', 'color': return_color}),
                html.H3(f"{daily_return:+.2%}", style={'margin': '10px 0 5px 0', 'color': return_color}),
                html.P("Daily Return", style={'margin': '0', 'color': COLORS['text']})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className="three columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'margin': '5px'
        })
    )
    
    # Active Positions Card
    cards.append(
        html.Div([
            html.Div([
                html.I(className="fas fa-list", style={'fontSize': '24px', 'color': COLORS['info']}),
                html.H3(f"{portfolio_data.get('num_positions', 0)}", style={'margin': '10px 0 5px 0'}),
                html.P("Active Positions", style={'margin': '0', 'color': COLORS['text']})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className="three columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'margin': '5px'
        })
    )
    
    # Risk Status Card
    risk_score = portfolio_data.get('risk_score', 0)
    risk_color = COLORS['success'] if risk_score < 50 else COLORS['warning'] if risk_score < 75 else COLORS['danger']
    
    cards.append(
        html.Div([
            html.Div([
                html.I(className="fas fa-shield-alt", style={'fontSize': '24px', 'color': risk_color}),
                html.H3(f"{risk_score:.0f}/100", style={'margin': '10px 0 5px 0', 'color': risk_color}),
                html.P("Risk Score", style={'margin': '0', 'color': COLORS['text']})
            ], style={'textAlign': 'center', 'padding': '20px'})
        ], className="three columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'margin': '5px'
        })
    )
    
    return html.Div(cards, className="row", style={'marginBottom': '20px'})

def create_portfolio_performance_tab():
    """Create portfolio performance tab content."""
    return html.Div([
        # Portfolio Value Chart
        html.Div([
            html.H3("Portfolio Value Over Time", style={'color': COLORS['primary']}),
            dcc.Graph(id='portfolio-value-chart')
        ], className="twelve columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'padding': '20px',
            'marginBottom': '20px'
        }),
        
        # Returns and Positions Row
        html.Div([
            # Monthly Returns Chart
            html.Div([
                html.H4("Monthly Returns", style={'color': COLORS['primary']}),
                dcc.Graph(id='monthly-returns-chart')
            ], className="six columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            }),
            
            # Current Positions Table
            html.Div([
                html.H4("Current Positions", style={'color': COLORS['primary']}),
                html.Div(id='positions-table')
            ], className="six columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            })
        ], className="row")
    ])

def create_trading_decisions_tab():
    """Create trading decisions tab content."""
    return html.Div([
        # Signal Heatmap
        html.Div([
            html.H3("Signal Strength Heatmap", style={'color': COLORS['primary']}),
            dcc.Graph(id='signal-heatmap')
        ], className="twelve columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'padding': '20px',
            'marginBottom': '20px'
        }),
        
        # Pending Trades and Trade History
        html.Div([
            # Pending Trades
            html.Div([
                html.H4("Pending Trades This Week", style={'color': COLORS['primary']}),
                html.Div(id='pending-trades')
            ], className="six columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            }),
            
            # Recent Trade History
            html.Div([
                html.H4("Recent Trade History", style={'color': COLORS['primary']}),
                html.Div(id='trade-history')
            ], className="six columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            })
        ], className="row")
    ])

def create_trade_approval_tab():
    """Create trade approval tab content."""
    return html.Div([
        # Trade Approval Header with Summary
        html.Div([
            html.Div([
                html.H3("Pending Trade Approvals", style={'color': COLORS['primary'], 'marginBottom': '10px'}),
                html.Div(id='approval-summary')
            ], className="eight columns"),
            
            html.Div([
                html.Button('Bulk Approve All', id='bulk-approve-btn', n_clicks=0, 
                           className='button button-primary', 
                           style={'margin': '5px', 'backgroundColor': COLORS['success']}),
                html.Button('Export PDF', id='export-pdf-btn', n_clicks=0,
                           className='button', 
                           style={'margin': '5px', 'backgroundColor': COLORS['info']}),
                html.Button('Send Notification', id='send-notification-btn', n_clicks=0,
                           className='button',
                           style={'margin': '5px', 'backgroundColor': COLORS['warning']})
            ], className="four columns", style={'textAlign': 'right', 'paddingTop': '20px'})
        ], className="row", style={'marginBottom': '20px'}),
        
        # Pending Trades Table
        html.Div([
            html.H4("Pending Trades", style={'color': COLORS['primary']}),
            html.Div(id='pending-trades-table'),
            
            # Trade Details Modal (initially hidden)
            html.Div(id='trade-details-modal', style={'display': 'none'})
        ], className="twelve columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'padding': '20px',
            'marginBottom': '20px'
        }),
        
        # Portfolio Impact Analysis
        html.Div([
            html.Div([
                html.H4("Portfolio Impact Analysis", style={'color': COLORS['primary']}),
                html.Div(id='portfolio-impact-chart')
            ], className="six columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            }),
            
            html.Div([
                html.H4("Action Log", style={'color': COLORS['primary']}),
                html.Div(id='approval-action-log')
            ], className="six columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            })
        ], className="row"),
        
        # Hidden div to store action results
        html.Div(id='approval-action-result', style={'display': 'none'})
    ])

def create_risk_metrics_tab():
    """Create risk metrics tab content."""
    return html.Div([
        # Risk Gauges Row
        html.Div([
            # Drawdown Gauge
            html.Div([
                html.H4("Current Drawdown", style={'color': COLORS['primary']}),
                dcc.Graph(id='drawdown-gauge')
            ], className="four columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            }),
            
            # Volatility Gauge
            html.Div([
                html.H4("Portfolio Volatility", style={'color': COLORS['primary']}),
                dcc.Graph(id='volatility-gauge')
            ], className="four columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            }),
            
            # Position Concentration Chart
            html.Div([
                html.H4("Position Concentration", style={'color': COLORS['primary']}),
                dcc.Graph(id='concentration-pie')
            ], className="four columns", style={
                'backgroundColor': COLORS['background'],
                'border': f'1px solid {COLORS["light"]}',
                'borderRadius': '8px',
                'padding': '20px'
            })
        ], className="row", style={'marginBottom': '20px'}),
        
        # Risk Alerts
        html.Div([
            html.H3("Risk Alerts", style={'color': COLORS['primary']}),
            html.Div(id='risk-alerts')
        ], className="twelve columns", style={
            'backgroundColor': COLORS['background'],
            'border': f'1px solid {COLORS["light"]}',
            'borderRadius': '8px',
            'padding': '20px'
        })
    ])

# Main layout
app.layout = html.Div([
    # Auto-refresh interval
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # 5 minutes in milliseconds
        n_intervals=0,
        disabled=not is_market_open()  # Only refresh when market is open
    ),
    
    # Header
    html.Div(id='header-container'),
    
    # Summary Cards
    html.Div(id='summary-cards'),
    
    # Main Content Tabs
    dcc.Tabs(id="main-tabs", value='portfolio', children=[
        dcc.Tab(label='Portfolio Performance', value='portfolio'),
        dcc.Tab(label='Trading Decisions', value='trading'),
        dcc.Tab(label='Trade Approval', value='approval'),
        dcc.Tab(label='Risk Metrics', value='risk')
    ], style={'marginBottom': '20px'}),
    
    # Tab Content
    html.Div(id='tab-content')
    
], style={
    'fontFamily': 'Arial, sans-serif',
    'margin': '0',
    'padding': '20px',
    'backgroundColor': '#f8f9fa'
})

# Callbacks
@app.callback(
    [Output('header-container', 'children'),
     Output('summary-cards', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_header_and_summary(n):
    """Update header and summary cards."""
    try:
        # Get portfolio data
        portfolio_data = get_portfolio_data()
        
        # Create header and summary cards
        header = create_header()
        summary = create_summary_cards(portfolio_data)
        
        return header, summary
    except Exception as e:
        logger.error(f"Error updating header: {e}")
        return create_header(), html.Div("Error loading summary data")

@app.callback(
    Output('tab-content', 'children'),
    [Input('main-tabs', 'value')]
)
def update_tab_content(active_tab):
    """Update tab content based on selected tab."""
    if active_tab == 'portfolio':
        return create_portfolio_performance_tab()
    elif active_tab == 'trading':
        return create_trading_decisions_tab()
    elif active_tab == 'approval':
        return create_trade_approval_tab()
    elif active_tab == 'risk':
        return create_risk_metrics_tab()
    else:
        return html.Div("Select a tab")

@app.callback(
    Output('portfolio-value-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_portfolio_chart(n):
    """Update portfolio value chart."""
    try:
        # Get portfolio snapshots
        snapshots = portfolio_manager.snapshots if portfolio_manager else []
        
        if not snapshots:
            # Create dummy data for demonstration
            dates = pd.date_range(start='2024-12-01', end='2025-01-31', freq='D')
            np.random.seed(42)
            values = 10000 * (1 + np.random.normal(0.0005, 0.015, len(dates))).cumprod()
            
            portfolio_df = pd.DataFrame({
                'date': dates,
                'value': values
            })
        else:
            portfolio_df = pd.DataFrame([
                {'date': pd.to_datetime(s.timestamp), 'value': s.total_value}
                for s in snapshots
            ])
        
        # Get benchmark data
        benchmark_df = get_benchmark_data()
        benchmark_df = benchmark_df[benchmark_df['date'] >= portfolio_df['date'].min()]
        
        # Create figure
        fig = go.Figure()
        
        # Portfolio line
        fig.add_trace(go.Scatter(
            x=portfolio_df['date'],
            y=portfolio_df['value'],
            mode='lines',
            name='Portfolio',
            line=dict(color=COLORS['primary'], width=3)
        ))
        
        # Benchmark line
        fig.add_trace(go.Scatter(
            x=benchmark_df['date'],
            y=benchmark_df['value'],
            mode='lines',
            name='S&P 500',
            line=dict(color=COLORS['warning'], width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio Performance vs S&P 500",
            xaxis_title="Date",
            yaxis_title="Value ($)",
            hovermode='x unified',
            template='plotly_white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating portfolio chart: {e}")
        return go.Figure().add_annotation(text="Error loading chart data", x=0.5, y=0.5)

@app.callback(
    Output('monthly-returns-chart', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_returns_chart(n):
    """Update monthly returns bar chart."""
    try:
        # Generate sample monthly returns data
        months = ['Oct 2024', 'Nov 2024', 'Dec 2024', 'Jan 2025']
        portfolio_returns = [0.025, -0.015, 0.045, 0.032]
        benchmark_returns = [0.018, -0.022, 0.038, 0.025]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=months,
            y=portfolio_returns,
            name='Portfolio',
            marker_color=COLORS['primary']
        ))
        
        fig.add_trace(go.Bar(
            x=months,
            y=benchmark_returns,
            name='S&P 500',
            marker_color=COLORS['warning']
        ))
        
        fig.update_layout(
            title="Monthly Returns Comparison",
            xaxis_title="Month",
            yaxis_title="Return (%)",
            yaxis=dict(tickformat='.1%'),
            barmode='group',
            template='plotly_white',
            height=300
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating returns chart: {e}")
        return go.Figure()

@app.callback(
    Output('positions-table', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_positions_table(n):
    """Update current positions table."""
    try:
        if not portfolio_manager:
            return html.P("Portfolio manager not initialized")
        
        positions_df = portfolio_manager.get_position_summary()
        
        if positions_df.empty:
            return html.P("No current positions")
        
        # Calculate P&L percentage
        positions_df['pnl_pct'] = (
            positions_df['unrealized_pnl'] / 
            (positions_df['shares'] * positions_df['avg_cost'])
        ).fillna(0) * 100
        
        # Format for display
        display_df = positions_df[['ticker', 'shares', 'market_value', 'weight', 'pnl_pct']].copy()
        display_df['shares'] = display_df['shares'].round(1)
        display_df['market_value'] = display_df['market_value'].round(0)
        display_df['weight'] = (display_df['weight'] * 100).round(1)
        display_df['pnl_pct'] = display_df['pnl_pct'].round(1)
        
        return dash_table.DataTable(
            data=display_df.to_dict('records'),
            columns=[
                {'name': 'Ticker', 'id': 'ticker'},
                {'name': 'Shares', 'id': 'shares', 'type': 'numeric'},
                {'name': 'Value ($)', 'id': 'market_value', 'type': 'numeric'},
                {'name': 'Weight (%)', 'id': 'weight', 'type': 'numeric'},
                {'name': 'P&L (%)', 'id': 'pnl_pct', 'type': 'numeric'}
            ],
            style_cell={'textAlign': 'center', 'fontSize': '12px'},
            style_data_conditional=[
                {
                    'if': {'filter_query': '{pnl_pct} > 0'},
                    'backgroundColor': '#d4edda',
                    'color': 'black',
                },
                {
                    'if': {'filter_query': '{pnl_pct} < 0'},
                    'backgroundColor': '#f8d7da',
                    'color': 'black',
                }
            ]
        )
        
    except Exception as e:
        logger.error(f"Error updating positions table: {e}")
        return html.P(f"Error: {str(e)}")

@app.callback(
    Output('signal-heatmap', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_signal_heatmap(n):
    """Update signal strength heatmap."""
    try:
        # Generate sample signal data
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD']
        signal_types = ['LLM Sentiment', 'Technical', 'Volatility', 'Sector']
        
        # Create sample data matrix
        np.random.seed(42)
        signal_matrix = np.random.uniform(0, 1, (len(tickers), len(signal_types)))
        
        fig = go.Figure(data=go.Heatmap(
            z=signal_matrix,
            x=signal_types,
            y=tickers,
            colorscale='RdYlGn',
            zmin=0,
            zmax=1,
            hovertemplate='%{y}<br>%{x}: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Signal Strength by Stock and Type",
            template='plotly_white',
            height=400
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error updating signal heatmap: {e}")
        return go.Figure()

@app.callback(
    [Output('drawdown-gauge', 'figure'),
     Output('volatility-gauge', 'figure'),
     Output('concentration-pie', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_risk_charts(n):
    """Update risk metric charts."""
    try:
        # Sample risk data
        current_drawdown = 0.123  # 12.3%
        portfolio_volatility = 0.285  # 28.5%
        
        # Drawdown gauge
        drawdown_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=current_drawdown * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Drawdown (%)"},
            gauge={
                'axis': {'range': [None, 25]},
                'bar': {'color': COLORS['danger']},
                'steps': [
                    {'range': [0, 10], 'color': COLORS['success']},
                    {'range': [10, 20], 'color': COLORS['warning']},
                    {'range': [20, 25], 'color': COLORS['danger']}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        drawdown_fig.update_layout(height=300)
        
        # Volatility gauge
        volatility_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=portfolio_volatility * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Volatility (%)"},
            gauge={
                'axis': {'range': [None, 50]},
                'bar': {'color': COLORS['info']},
                'steps': [
                    {'range': [0, 20], 'color': COLORS['success']},
                    {'range': [20, 35], 'color': COLORS['warning']},
                    {'range': [35, 50], 'color': COLORS['danger']}
                ]
            }
        ))
        volatility_fig.update_layout(height=300)
        
        # Position concentration pie chart
        sectors = ['Semiconductors', 'Software', 'E-commerce', 'Social Media', 'Other']
        weights = [0.35, 0.25, 0.20, 0.15, 0.05]
        
        concentration_fig = go.Figure(data=[go.Pie(
            labels=sectors,
            values=weights,
            hole=0.4,
            marker=dict(colors=[COLORS['primary'], COLORS['info'], COLORS['success'], COLORS['warning'], COLORS['danger']])
        )])
        concentration_fig.update_layout(height=300, showlegend=True)
        
        return drawdown_fig, volatility_fig, concentration_fig
        
    except Exception as e:
        logger.error(f"Error updating risk charts: {e}")
        return go.Figure(), go.Figure(), go.Figure()

def get_portfolio_data() -> Dict[str, Any]:
    """Get current portfolio data."""
    try:
        if not portfolio_manager:
            return {
                'total_value': 10000,
                'daily_return': 0.0,
                'num_positions': 0,
                'risk_score': 0
            }
        
        # Get latest metrics
        metrics = portfolio_manager.calculate_performance_metrics()
        positions_df = portfolio_manager.get_position_summary()
        
        # Calculate daily return (simplified)
        daily_return = 0.0
        if len(portfolio_manager.snapshots) >= 2:
            latest = portfolio_manager.snapshots[-1]
            previous = portfolio_manager.snapshots[-2]
            daily_return = (latest.total_value / previous.total_value) - 1
        
        return {
            'total_value': portfolio_manager.total_value,
            'daily_return': daily_return,
            'num_positions': len(positions_df) if not positions_df.empty else 0,
            'risk_score': 45  # Sample risk score
        }
        
    except Exception as e:
        logger.error(f"Error getting portfolio data: {e}")
        return {
            'total_value': 0,
            'daily_return': 0.0,
            'num_positions': 0,
            'risk_score': 0
        }

# Trade Approval Callbacks

@app.callback(
    [Output('approval-summary', 'children'),
     Output('pending-trades-table', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('approval-action-result', 'children')]
)
def update_trade_approval_content(n, action_result):
    """Update trade approval tab content."""
    try:
        if not trade_approval_system:
            return html.P("Trade approval system not initialized"), html.P("System not available")
        
        # Create sample pending trades for demonstration
        if len(trade_approval_system.pending_trades) == 0:
            create_sample_pending_trades()
        
        pending_trades = trade_approval_system.get_pending_trades()
        impact_summary = trade_approval_system.get_portfolio_impact_summary()
        
        # Create approval summary
        summary = html.Div([
            html.P([
                html.Strong(f"{impact_summary['total_trades']} pending trades"), 
                f" • {impact_summary['buy_count']} buys, {impact_summary['sell_count']} sells"
            ]),
            html.P([
                f"Net capital change: ", 
                html.Span(f"${impact_summary['net_capital_change']:+,.0f}", 
                         style={'color': COLORS['success'] if impact_summary['net_capital_change'] >= 0 else COLORS['danger']}),
                f" • High confidence: {impact_summary['high_confidence_trades']} trades"
            ])
        ])
        
        # Create pending trades table
        if not pending_trades:
            trades_table = html.P("No pending trades")
        else:
            # Prepare data for table
            table_data = []
            for trade in pending_trades:
                table_data.append({
                    'ticker': trade.ticker,
                    'action': trade.action.value,
                    'shares': f"{trade.current_shares:.0f}",
                    'value': f"${trade.estimated_value:,.0f}",
                    'confidence': f"{trade.confidence_score:.1%}",
                    'signal': f"{trade.signal_strength:.3f}",
                    'reasoning': trade.reasoning[:60] + "..." if len(trade.reasoning) > 60 else trade.reasoning,
                    'trade_id': trade.trade_id
                })
            
            # Create interactive table with action buttons
            table_rows = []
            
            # Header row
            header_row = html.Tr([
                html.Th("Ticker"),
                html.Th("Action"),
                html.Th("Shares"),
                html.Th("Value"),
                html.Th("Confidence"),
                html.Th("Signal"),
                html.Th("Reasoning"),
                html.Th("Actions")
            ], style={'backgroundColor': COLORS['primary'], 'color': 'white'})
            table_rows.append(header_row)
            
            # Data rows
            for i, data in enumerate(table_data):
                action_color = COLORS['success'] if data['action'] == 'BUY' else COLORS['danger']
                confidence_num = float(data['confidence'].strip('%')) / 100
                confidence_color = COLORS['success'] if confidence_num > 0.7 else COLORS['warning'] if confidence_num > 0.5 else COLORS['danger']
                
                row = html.Tr([
                    html.Td(data['ticker'], style={'fontWeight': 'bold'}),
                    html.Td(data['action'], style={'color': action_color, 'fontWeight': 'bold'}),
                    html.Td(data['shares']),
                    html.Td(data['value']),
                    html.Td(data['confidence'], style={'color': confidence_color}),
                    html.Td(data['signal']),
                    html.Td(data['reasoning'], style={'fontSize': '12px'}),
                    html.Td([
                        html.Button('✓', id={'type': 'approve-btn', 'index': i}, n_clicks=0,
                                   style={'backgroundColor': COLORS['success'], 'color': 'white', 'border': 'none', 'margin': '2px', 'borderRadius': '4px'}),
                        html.Button('✗', id={'type': 'reject-btn', 'index': i}, n_clicks=0,
                                   style={'backgroundColor': COLORS['danger'], 'color': 'white', 'border': 'none', 'margin': '2px', 'borderRadius': '4px'}),
                        html.Button('📝', id={'type': 'modify-btn', 'index': i}, n_clicks=0,
                                   style={'backgroundColor': COLORS['info'], 'color': 'white', 'border': 'none', 'margin': '2px', 'borderRadius': '4px'})
                    ])
                ], style={'backgroundColor': '#f9f9f9' if i % 2 == 0 else 'white'})
                table_rows.append(row)
            
            trades_table = html.Table(table_rows, style={'width': '100%', 'borderCollapse': 'collapse'})
        
        return summary, trades_table
        
    except Exception as e:
        logger.error(f"Error updating trade approval content: {e}")
        return html.P(f"Error: {str(e)}"), html.P("Error loading trades")

@app.callback(
    Output('portfolio-impact-chart', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_portfolio_impact_chart(n):
    """Update portfolio impact visualization."""
    try:
        if not trade_approval_system:
            return html.P("System not available")
        
        pending_trades = trade_approval_system.get_pending_trades()
        
        if not pending_trades:
            return html.P("No pending trades to analyze")
        
        # Create sector impact chart
        sector_impact = {}
        for trade in pending_trades:
            sector = trade.sector
            impact = trade.estimated_value if trade.action == TradeAction.BUY else -trade.estimated_value
            
            if sector not in sector_impact:
                sector_impact[sector] = 0
            sector_impact[sector] += impact
        
        sectors = list(sector_impact.keys())
        impacts = list(sector_impact.values())
        colors_list = [COLORS['success'] if x >= 0 else COLORS['danger'] for x in impacts]
        
        fig = go.Figure(data=[
            go.Bar(x=sectors, y=impacts, marker_color=colors_list)
        ])
        
        fig.update_layout(
            title="Net Capital Impact by Sector",
            xaxis_title="Sector",
            yaxis_title="Net Impact ($)",
            template='plotly_white',
            height=300
        )
        
        return dcc.Graph(figure=fig)
        
    except Exception as e:
        logger.error(f"Error updating portfolio impact chart: {e}")
        return html.P(f"Error: {str(e)}")

@app.callback(
    Output('approval-action-log', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_action_log(n):
    """Update approval action log."""
    try:
        # Create sample action log
        actions = [
            {'time': '14:30:22', 'action': 'Trade AAPL-BUY created', 'type': 'info'},
            {'time': '14:25:15', 'action': 'Trade TSLA-SELL approved', 'type': 'success'},
            {'time': '14:20:08', 'action': 'Trade NVDA-BUY modified (150→100 shares)', 'type': 'warning'},
            {'time': '14:15:33', 'action': 'Bulk approval completed (3 trades)', 'type': 'success'},
            {'time': '14:10:45', 'action': 'PDF report exported', 'type': 'info'}
        ]
        
        log_items = []
        for action in actions:
            color = {
                'success': COLORS['success'],
                'warning': COLORS['warning'],
                'danger': COLORS['danger'],
                'info': COLORS['info']
            }.get(action['type'], COLORS['text'])
            
            log_items.append(
                html.Div([
                    html.Span(action['time'], style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Span(action['action'], style={'color': color})
                ], style={'padding': '5px 0', 'borderBottom': '1px solid #eee'})
            )
        
        return html.Div(log_items, style={'maxHeight': '250px', 'overflowY': 'auto'})
        
    except Exception as e:
        logger.error(f"Error updating action log: {e}")
        return html.P(f"Error: {str(e)}")

# Button callbacks for trade actions
@app.callback(
    Output('approval-action-result', 'children'),
    [Input('bulk-approve-btn', 'n_clicks'),
     Input('export-pdf-btn', 'n_clicks'),
     Input('send-notification-btn', 'n_clicks')]
)
def handle_bulk_actions(bulk_approve_clicks, export_clicks, notification_clicks):
    """Handle bulk action buttons."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return ""
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    try:
        if button_id == 'bulk-approve-btn' and bulk_approve_clicks > 0:
            if trade_approval_system:
                pending_trades = trade_approval_system.get_pending_trades()
                trade_ids = [t.trade_id for t in pending_trades]
                approved_count = trade_approval_system.bulk_approve_trades(trade_ids)
                return f"Bulk approved {approved_count} trades"
            
        elif button_id == 'export-pdf-btn' and export_clicks > 0:
            if trade_approval_system:
                try:
                    pdf_path = trade_approval_system.generate_weekly_summary_pdf()
                    return f"PDF exported to {pdf_path}"
                except ImportError:
                    return "PDF export requires reportlab package: pip install reportlab"
                
        elif button_id == 'send-notification-btn' and notification_clicks > 0:
            if trade_approval_system:
                success = trade_approval_system.notify_pending_trades()
                return "Notification sent" if success else "Notification failed (check email config)"
    
    except Exception as e:
        logger.error(f"Error handling bulk action: {e}")
        return f"Error: {str(e)}"
    
    return ""

def create_sample_pending_trades():
    """Create sample pending trades for demonstration."""
    if not trade_approval_system:
        return
    
    try:
        sample_trades = [
            {
                'ticker': 'AAPL',
                'action': TradeAction.BUY,
                'shares': 100,
                'price': 225.50,
                'reasoning': 'Strong earnings momentum and positive analyst upgrades driving bullish sentiment',
                'llm_analysis': 'LLM Analysis: Bullish sentiment detected with 85% confidence. Recent news shows positive iPhone sales data and services growth acceleration.',
                'confidence_score': 0.85,
                'signal_strength': 0.78,
                'portfolio_impact': {'weight_change': 0.05, 'risk_description': 'Low risk addition to core holding'}
            },
            {
                'ticker': 'TSLA',
                'action': TradeAction.SELL,
                'shares': 50,
                'price': 195.40,
                'reasoning': 'Trailing stop-loss triggered at 15% below peak price for risk management',
                'llm_analysis': 'LLM Analysis: Risk management sell signal with 95% confidence. Stop-loss mechanism activated due to price decline.',
                'confidence_score': 0.95,
                'signal_strength': 0.92,
                'portfolio_impact': {'weight_change': -0.03, 'risk_description': 'Risk reduction through stop-loss'}
            },
            {
                'ticker': 'NVDA',
                'action': TradeAction.BUY,
                'shares': 75,
                'price': 875.25,
                'reasoning': 'AI infrastructure demand surge and strong datacenter revenue growth outlook',
                'llm_analysis': 'LLM Analysis: Very bullish on AI infrastructure play. High conviction trade with 90% confidence based on earnings and guidance.',
                'confidence_score': 0.90,
                'signal_strength': 0.88,
                'portfolio_impact': {'weight_change': 0.08, 'risk_description': 'High conviction growth play'}
            }
        ]
        
        for trade_data in sample_trades:
            trade_approval_system.create_pending_trade(**trade_data)
            
        logger.info("Created sample pending trades")
        
    except Exception as e:
        logger.error(f"Error creating sample trades: {e}")

def main():
    """Run the dashboard server."""
    print("Initializing trading system components...")
    initialize_components()
    
    print("Starting dashboard server...")
    print("Dashboard will be available at: http://127.0.0.1:8050")
    print("Press Ctrl+C to stop the server")
    
    # Run the app
    app.run_server(
        debug=False,  # Set to False for production
        host='127.0.0.1',
        port=8050,
        dev_tools_hot_reload=False  # Disable hot reload for stability
    )

if __name__ == '__main__':
    main()