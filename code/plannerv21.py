import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from pathlib import Path
from scipy.optimize import minimize
from dash.exceptions import PreventUpdate
from chatbot import query_ollama
import time

# ------------------ Load Preprocessed Data ------------------ #
data_dir = Path("data/processed")
returns_df = pd.read_csv(data_dir / "cleaned_returns.csv", parse_dates=["date"])
returns_df.sort_values("date", inplace=True)
assets = ["stock_excess", "bond_excess"]

# ------------------ Session Memory ------------------ #
latest_simulation_summary = ""
chat_history = []

# ------------------ Optimization Functions ------------------ #

def sharpe_optimal_weights(expected_returns, cov_matrix, risk_free_rate):
    n = len(expected_returns)
    def neg_sharpe(weights):
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -1 * (port_return - risk_free_rate) / port_vol
    result = minimize(neg_sharpe, np.ones(n)/n, bounds=[(0,1)]*n, constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    return result.x

def min_variance_weights(cov_matrix):
    n = len(cov_matrix)
    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    result = minimize(portfolio_volatility, np.ones(n)/n, bounds=[(0,1)]*n, constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    return result.x

# ------------------ Glidepath Functions ------------------ #

def define_dynamic_stages(total_years):
    return {
        'young': (0.9, int(0.5 * total_years)),
        'transition': (0.6, int(0.75 * total_years)),
        'early': (0.3, int(0.9 * total_years)),
        'late': (0.1, total_years)
    }

def generate_data_informed_glidepath(n_years, returns_df, floor_dict):
    monthly_returns = returns_df.set_index("date")[assets]
    monthly_mean = monthly_returns.mean() * 12
    monthly_cov = monthly_returns.cov() * 12
    glidepath = []
    for year in range(n_years):
        x = year / n_years
        equity = 0.1 + (0.8 / (1 + np.exp(10 * (x - 0.5))))
        glidepath.append([max(min(equity, 0.9), 0.1), 1 - max(min(equity, 0.9), 0.1)])
    return [w[0] for w in glidepath], [w[1] for w in glidepath]

# ------------------ Simulation ------------------ #

def simulate_retirement_plan(income, savings_rate, years, init_savings, strategy="standard", n_sim=1000, monthly_withdrawal=0):
    total_years = years + 20
    cma_means = np.array([0.06, 0.025])
    cma_cov = np.array([[0.0225, 0.00225], [0.00225, 0.0025]])
    all_terminal_values = []
    floor_dict = define_dynamic_stages(total_years)

    if strategy == "simulated":
        stock_path, bond_path = generate_data_informed_glidepath(total_years, returns_df, floor_dict)
    else:
        stock_path = [0.1 + (0.8 / (1 + np.exp(10 * (year / total_years - 0.5)))) for year in range(total_years)]
        bond_path = [1 - sw for sw in stock_path]

    for sim in range(n_sim):
        savings = init_savings
        projected = []
        returns = np.random.multivariate_normal(cma_means, cma_cov, total_years)

        for year in range(total_years):
            equity_ret, bond_ret = returns[year]
            alloc = np.array([stock_path[year], bond_path[year]])
            portfolio_ret = alloc[0] * equity_ret + alloc[1] * bond_ret
            savings = savings * (1 + portfolio_ret) + (income * savings_rate if year < years else 0) - (12 * monthly_withdrawal if year >= years else 0)
            projected.append(max(savings, 0))

        all_terminal_values.append(projected)

    all_terminal_values = np.array(all_terminal_values)
    return np.median(all_terminal_values, axis=0).tolist(), np.percentile(all_terminal_values, 10, axis=0).tolist(), np.percentile(all_terminal_values, 90, axis=0).tolist(), stock_path, bond_path

# ==================== Dash App Setup ==================== #
external_stylesheets = [
    'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "AI‑Assisted Retirement Planner"

# ==================== Layout Helpers ==================== #
COLORS = {
    'background': '#12151A',
    'card_bg':   '#1E2126',
    'input_bg':  '#2C3037',
    'primary':   '#4C9AFF',
    'primary_hover': '#3D7CC7',
    'text':      '#FFFFFF',
    'text_muted':'#B7BEC9',
    'border':    '#2C3037',
}

INPUT_STYLE  = {
    'width': '100%', 'backgroundColor': COLORS['input_bg'], 'color': COLORS['text'],
    'borderRadius': '8px', 'border': f'1px solid {COLORS["border"]}',
    'padding': '12px', 'fontSize': '14px', 'marginBottom': '16px', 'boxSizing': 'border-box'
}
LABEL_STYLE  = {'color': COLORS['text'], 'fontWeight': '600', 'marginBottom': '8px',
               'display': 'block', 'fontSize': '14px'}
SLIDER_STYLE = {'marginBottom': '24px'}
BUTTON_STYLE = {
    'width': '100%', 'backgroundColor': COLORS['primary'], 'color': COLORS['text'],
    'border': 'none', 'borderRadius': '8px', 'padding': '14px',
    'fontWeight': '600', 'fontSize': '16px', 'cursor': 'pointer',
    'transition': 'background-color 0.2s ease', 'marginTop': '8px', 'marginBottom': '24px'
}

# ==================== Main Layout ==================== #
app.layout = html.Div([
    # ---------- MAIN ROW ----------
    html.Div([
        # ===== Left panel (inputs) =====
        html.Div([
            html.Div([
                html.Label("Annual Net Income ($)", style=LABEL_STYLE),
                dcc.Input(id='income', type='number', value=75000, style=INPUT_STYLE),

                html.Label("Annual Savings Rate (%)", style=LABEL_STYLE),
                dcc.Slider(
                    id='savings_rate', min=0, max=100, step=1, value=15,
                    marks={i: {'label': f'{i}%', 'style': {'color': COLORS['text_muted']}}
                           for i in range(0, 101, 10)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag', className='custom-slider'
                ),

                html.Label("Years Until Retirement", style=LABEL_STYLE),
                dcc.Slider(
                    id='years', min=1, max=50, step=1, value=30,
                    marks={i: {'label': str(i), 'style': {'color': COLORS['text_muted']}}
                           for i in range(0, 51, 5)},
                    tooltip={"placement": "bottom", "always_visible": True},
                    updatemode='drag', className='custom-slider'
                ),

                html.Label("Initial Retirement Savings ($)", style=LABEL_STYLE),
                dcc.Input(id='init_savings', type='number', value=10000, style=INPUT_STYLE),

                html.Label("Desired Monthly Withdrawal in Retirement ($)", style=LABEL_STYLE),
                dcc.Input(id='monthly_withdrawal', type='number', value=0, style=INPUT_STYLE),

                html.Label("Glidepath Strategy", style=LABEL_STYLE),
                dcc.RadioItems(
                    id='strategy',
                    options=[
                        {'label': 'Industry Standard',   'value': 'standard'},
                        {'label': 'Simulated Optimized', 'value': 'simulated'}
                    ],
                    value='standard',
                    labelStyle={'display': 'flex', 'alignItems': 'center', 'color': COLORS['text'],
                               'marginBottom': '8px', 'cursor': 'pointer', 'fontSize': '14px'},
                    style={'marginBottom': '24px'}
                ),

                html.Button('Generate Plan', id='submit-button', n_clicks=0, style=BUTTON_STYLE)
            ], style={'padding': '24px', 'boxSizing': 'border-box', 'overflowY': 'auto',
                      'maxHeight': 'calc(100vh - 64px)'}),
        ], style={'flex': '0 0 350px', 'backgroundColor': COLORS['card_bg'],
                  'borderRadius': '12px', 'boxShadow': '0 8px 24px rgba(0,0,0,0.12)',
                  'marginRight': '24px', 'maxHeight': 'calc(100vh - 64px)', 'overflowY': 'auto'}),

        # ===== Right panel (header + graph) =====
        html.Div([
            # --- Header / summary row ---
            html.Div([
                html.Div("AI‑Assisted Retirement Planner", style={
                    "fontFamily": "Inter, sans-serif", "color": COLORS['text'],
                    "fontWeight": "700", "fontSize": "24px"}),
                html.Div([
                    html.Span("Median Value at Retirement: ", style={'fontWeight': '600', 'color': COLORS['text_muted']}),
                    html.Span(id="retirement-value", style={'color': COLORS['text'], 'fontWeight': '600'}),
                    html.Br(),
                    html.Span("Median Value After 20 Years: ", style={'fontWeight': '600', 'color': COLORS['text_muted']}),
                    html.Span(id="twenty-year-value", style={'color': COLORS['text'], 'fontWeight': '600'})
                ], style={'textAlign': 'right'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                      'backgroundColor': COLORS['card_bg'], 'padding': '20px', 'borderRadius': '12px',
                      'color': COLORS['text'], 'marginBottom': '20px',
                      'boxShadow': '0 4px 12px rgba(0,0,0,0.08)'}),

            # --- Graph container (flex‑grows) ---
            dcc.Graph(
                id='retirement-graph', config={'displayModeBar': False},
                style={'flex': '1 1 auto', 'backgroundColor': COLORS['card_bg'],
                       'borderRadius': '12px', 'padding': '12px',
                       'boxShadow': '0 4px 12px rgba(0,0,0,0.08)'}
            )
        ], style={'flex': '1', 'height': 'calc(100vh - 64px)', 'display': 'flex',
                  'flexDirection': 'column', 'overflow': 'hidden'})
    ], style={'display': 'flex', 'padding': '32px', 'maxWidth': '1800px',
              'margin': '0 auto'}),

    # ===== Floating Chat Button =====
    html.Button([
        "\uD83D\uDCAC ", html.Span("Chat with AI", style={'marginLeft': '6px'})
    ], id='open-chatbot', n_clicks=0,
       style={'position': 'fixed', 'bottom': '24px', 'right': '24px', 'padding': '14px 20px',
              'fontSize': '16px', 'backgroundColor': COLORS['primary'], 'color': COLORS['text'],
              'border': 'none', 'borderRadius': '50px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.25)',
              'cursor': 'pointer', 'display': 'flex', 'alignItems': 'center', 'zIndex': '1001'}),

    # ===== Chatbot Modal =====
    html.Div(id='chatbot-modal', style={
        'display': 'none', # hidden until user clicks button
        'position': 'fixed', 'bottom': '90px', 'right': '24px',
        'width': '380px', 'height': '500px', 'backgroundColor': COLORS['card_bg'],
        'borderRadius': '12px', 'boxShadow': '0 12px 24px rgba(0,0,0,0.3)',
        'fontFamily': 'Inter, sans-serif', 'zIndex': '1000',
        'border': f'1px solid {COLORS["border"]}',
        'display': 'flex', 'flexDirection': 'column', 'overflow': 'hidden'
    }, children=[
        html.Div([
            html.H3("Retirement AI Assistant", style={'margin': '0', 'color': COLORS['text'], 'fontWeight': '600', 'fontSize': '16px'}),
            html.Button('✖', id='close-chatbot', style={'background': 'none', 'border': 'none', 'fontSize': '18px', 'cursor': 'pointer', 'color': COLORS['text_muted']})
        ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'padding': '16px', 'borderBottom': f'1px solid {COLORS["border"]}'}),

        # -------- Scrollable message area --------
        html.Div(id='chatbot-response', style={'padding': '16px', 'overflowY': 'auto', 'overflowX': 'hidden',
                                               'flex': '1', 'color': COLORS['text'], 'wordBreak': 'break-word',
                                               'whiteSpace': 'pre-wrap'}),

        # Input row
        html.Div([
            dcc.Input(id='user-question', type='text', placeholder='Ask a question about your retirement plan...',
                      style={'flex': '1', 'backgroundColor': COLORS['input_bg'], 'color': COLORS['text'],
                             'border': f'1px solid {COLORS["border"]}', 'borderRadius': '8px', 'padding': '12px', 'fontSize': '14px'}),
            html.Button('Ask', id='ask-button', n_clicks=0,
                       style={'marginLeft': '8px', 'backgroundColor': COLORS['primary'], 'color': COLORS['text'],
                              'border': 'none', 'borderRadius': '8px', 'padding': '12px 16px', 'fontWeight': '600', 'cursor': 'pointer'})
        ], style={'padding': '16px', 'borderTop': f'1px solid {COLORS["border"]}',
                  'display': 'flex', 'alignItems': 'center'}),
        # Hidden loading overlay (absolute)
        html.Div([
            dcc.Loading(id='loading-chatbot', type='circle', color=COLORS['primary'], children=[html.Div(id='loading-output')])
        ], style={'position': 'absolute', 'top': '50%', 'left': '50%',
                  'transform': 'translate(-50%, -50%)', 'zIndex': '1002'})
    ])
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'fontFamily': 'Inter, sans-serif'})

# ==================== Callbacks ==================== #
# --- Simulation / chart callback ---
@app.callback(
    [Output('retirement-graph', 'figure'), Output('retirement-value', 'children'), Output('twenty-year-value', 'children')],
    Input('submit-button', 'n_clicks'),
    State('income', 'value'), State('savings_rate', 'value'), State('years', 'value'),
    State('init_savings', 'value'), State('monthly_withdrawal', 'value'), State('strategy', 'value'))
def update_plan(n_clicks, income, savings_rate, years, init_savings, monthly_withdrawal, strategy):
    if not n_clicks:
        raise PreventUpdate

    projected, lower, upper, stock_wt, bond_wt = simulate_retirement_plan(
        income, savings_rate/100, years, init_savings, strategy, monthly_withdrawal=monthly_withdrawal)

    total_years = years + 20
    x = list(range(1, total_years + 1))

    fig = go.Figure([
        go.Scatter(x=x, y=stock_wt, mode='none', stackgroup='one', name='Equity Allocation (%)',
                    yaxis='y2', fillcolor='rgba(0,123,255,0.5)'),
        go.Scatter(x=x, y=bond_wt, mode='none', stackgroup='one', name='Bond Allocation (%)',
                    yaxis='y2', fillcolor='rgba(220,53,69,0.5)'),
        go.Scatter(x=x, y=upper, mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'),
        go.Scatter(x=x, y=lower, mode='lines', line=dict(width=0), fill='tonexty', name='10-90% Range',
                    fillcolor='rgba(0,200,255,0.2)', hoverinfo='skip'),
        go.Scatter(x=x, y=projected, mode='lines+markers', name='Median Portfolio Value',
                    line=dict(color='white', width=3), marker=dict(size=4, color='white'), yaxis='y1')
    ])

    caption = "Dynamic Glidepath Projection (Standard)" if strategy == 'standard' else "Dynamic Glidepath Projection (Simulated Optimized)"
    fig.update_layout(title={'text': caption, 'font': {'color': 'white', 'size': 20}, 'x': 0.5},
                      plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E',
                      font=dict(color='white', family='Inter, sans-serif'),
                      xaxis=dict(title='Years', gridcolor='#333', zerolinecolor='#333'),
                      yaxis=dict(title='Portfolio Value ($)', gridcolor='#333', zerolinecolor='#333'),
                      yaxis2=dict(title='Allocation %', overlaying='y', side='right', range=[0,1], tickformat='.0%', gridcolor='#333'),
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

    retire_val = f"${projected[years-1]:,.2f}"
    twenty_val = f"${projected[-1]:,.2f}"
    return fig, retire_val, twenty_val

# --- Chatbot modal visibility ---
@app.callback(Output('chatbot-modal', 'style'),
              [Input('open-chatbot', 'n_clicks'), Input('close-chatbot', 'n_clicks')],
              State('chatbot-modal', 'style'))
def toggle_chatbot(open_clicks, close_clicks, style):
    if style is None:
        style = {'display': 'none'}
    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    style['display'] = 'block' if trigger == 'open-chatbot' else 'none'
    return style

# --- Chatbot conversation ---
@app.callback(Output('chatbot-response', 'children'), Input('ask-button', 'n_clicks'), State('user-question', 'value'))
def update_chatbot(n_clicks, question):
    global chat_history, latest_simulation_summary
    if not n_clicks or not question:
        raise PreventUpdate

    chat_history.append(f"User: {question}")
    time.sleep(0.5)
    prompt = f"{latest_simulation_summary}\n\n" + "\n".join(chat_history) + "\n\nAssistant:"
    answer = query_ollama(prompt)
    chat_history.append(f"Assistant: {answer}")

    bubbles = []
    for turn in chat_history:
        role, text = turn.split(":", 1)
        align = 'left' if role.strip() == 'Assistant' else 'right'
        color = '#f1f1f1' if role.strip() == 'Assistant' else '#0074D9'
        text_color = 'black' if role.strip() == 'Assistant' else 'white'
        bubbles.append(dcc.Markdown(text.strip(), style={'backgroundColor': color, 'color': text_color,
                                                         'padding': '8px 12px', 'borderRadius': '16px', 'margin': '8px',
                                                         'textAlign': align, 'maxWidth': '80%',
                                                         'marginLeft': 'auto' if align=='right' else 'initial',
                                                         'marginRight': 'auto' if align=='left' else 'initial'}))
    bubbles.append(html.Div(id='scroll-target'))
    return bubbles

if __name__ == '__main__':
    app.run(debug=True)