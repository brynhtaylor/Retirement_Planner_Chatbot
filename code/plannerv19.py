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

external_stylesheets = ['https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600&display=swap']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "AI-Assisted Retirement Planner"

# ==================== Layout ==================== #

app.layout = html.Div([

    html.Div([

        html.Div([
            html.Div([
                html.H1("AI-Assisted Retirement Planner", style={
                    "textAlign": "center", "fontFamily": "Open Sans, sans-serif", "color": "#FFFFFF",
                    "paddingTop": "20px", "fontWeight": "bold", "fontSize": "30px"
                }),

                html.Label("Annual Net Income ($)", style={"color": "white", "fontWeight": "600", "marginBottom": "5px"}),
                dcc.Input(id='income', type='number', value=75000, style={
                    'width': '90%', 'backgroundColor': '#40444B', 'color': 'white', 'borderRadius': '8px',
                    'padding': '8px', 'border': 'none', 'marginBottom': '10px'
                }),

                html.Label("Annual Savings Rate (%)", style={"color": "white", "fontWeight": "600", "marginBottom": "5px"}),
                dcc.Slider(id='savings_rate', min=0, max=100, step=1, value=15, marks={i: f'{i}%' for i in range(0, 101, 10)}, 
                    tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'
                ),

                html.Label("Years Until Retirement", style={"color": "white", "fontWeight": "600", "marginBottom": "5px", "marginTop": "10px"}),
                dcc.Slider(id='years', min=1, max=50, step=1, value=30, marks={i: str(i) for i in range(0, 51, 5)}, 
                    tooltip={"placement": "bottom", "always_visible": True}, updatemode='drag'
                ),

                html.Label("Initial Retirement Savings ($)", style={"color": "white", "fontWeight": "600", "marginBottom": "5px", "marginTop": "10px"}),
                dcc.Input(id='init_savings', type='number', value=10000, style={
                    'width': '90%', 'backgroundColor': '#40444B', 'color': 'white', 'borderRadius': '8px',
                    'padding': '8px', 'border': 'none', 'marginBottom': '10px'
                }),

                html.Label("Desired Monthly Withdrawal in Retirement ($)", style={"color": "white", "fontWeight": "600", "marginBottom": "5px"}),
                dcc.Input(id='monthly_withdrawal', type='number', value=0, style={
                    'width': '90%', 'backgroundColor': '#40444B', 'color': 'white', 'borderRadius': '8px',
                    'padding': '8px', 'border': 'none', 'marginBottom': '15px'
                }),

                html.Label("Glidepath Strategy", style={"color": "white", "fontWeight": "600", "marginBottom": "5px"}),
                dcc.RadioItems(id='strategy', options=[
                    {'label': 'Industry Standard', 'value': 'standard'},
                    {'label': 'Simulated Optimized', 'value': 'simulated'}
                ], value='standard', labelStyle={'display': 'block', 'color': 'white', 'marginBottom': '5px'}),

                html.Button('Generate Plan', id='submit-button', n_clicks=0, style={
                    'width': '90%', 'backgroundColor': '#0074D9', 'color': 'white', 'border': 'none',
                    'borderRadius': '8px', 'padding': '12px', 'marginTop': '20px', 'fontWeight': 'bold', 'cursor': 'pointer'
                })

            ], style={'padding': '20px'})
        ], style={
            'flex': '0 0 30%', 'backgroundColor': '#2C2F33', 'borderRadius': '12px',
            'boxShadow': '0 8px 16px rgba(0,0,0,0.4)', 'minWidth': '300px', 'marginRight': '20px'
        }),

        html.Div([
            html.Div(id='summary-output', style={'color': 'white', 'fontSize': '16px', 'marginBottom': '15px'}),
            dcc.Graph(id='retirement-graph', config={'displayModeBar': False}, style={'height': '90vh'})
        ], style={'flex': '1', 'padding': '20px', 'height': '100%'})

    ], style={
        'display': 'flex', 'flexDirection': 'row', 'alignItems': 'stretch', 'justifyContent': 'center', 'padding': '20px',
        'height': 'calc(100vh - 80px)', 'gap': '20px'
    }),

    # Floating Chat Button
    html.Button("\ud83d\udcac Chat with AI", id="open-chatbot", n_clicks=0, style={
        'position': 'fixed', 'bottom': '20px', 'right': '20px', 'padding': '15px 20px',
        'fontSize': '18px', 'backgroundColor': '#0074D9', 'color': 'white', 'border': 'none',
        'borderRadius': '50px', 'boxShadow': '0 4px 8px rgba(0,0,0,0.2)', 'cursor': 'pointer'
    }),

    # Chatbot Modal
    html.Div(id='chatbot-modal', style={
        'display': 'none', 'position': 'fixed', 'bottom': '80px', 'right': '20px',
        'width': '350px', 'height': '450px', 'backgroundColor': '#2C2F33', 'borderRadius': '12px',
        'boxShadow': '0 8px 16px rgba(0,0,0,0.4)', 'overflowY': 'auto', 'padding': '10px', 'fontFamily': 'Open Sans, sans-serif'
    }, children=[
        html.Button('\u2716', id='close-chatbot', style={
            'background': 'none', 'border': 'none', 'fontSize': '20px', 'float': 'right', 'cursor': 'pointer', 'color': 'white'
        }),
        html.Div(id='chatbot-response', style={'marginBottom': '10px', 'maxHeight': '350px', 'overflowY': 'auto'}),
        dcc.Input(id='user-question', type='text', placeholder='Ask a question...', style={
            'width': '68%', 'backgroundColor': '#40444B', 'color': 'white', 'border': 'none', 'borderRadius': '6px',
            'padding': '8px', 'fontSize': '14px'
        }),
        html.Button('Ask', id='ask-button', n_clicks=0, style={
            'marginLeft': '10px', 'backgroundColor': '#0074D9', 'color': 'white', 'border': 'none',
            'borderRadius': '6px', 'padding': '8px 16px', 'fontWeight': 'bold', 'cursor': 'pointer'
        }),
        dcc.Loading(id='loading-chatbot', type='circle', children=[html.Div(id='loading-output')])
    ])

], style={
    'backgroundColor': '#1E1E1E', 'minHeight': '100vh', 'fontFamily': 'Open Sans, sans-serif'
})

# ==================== End Layout ==================== #

# ------------------ Callbacks ------------------ #

@app.callback(
    [Output('retirement-graph', 'figure'),
     Output('summary-output', 'children')],
    Input('submit-button', 'n_clicks'),
    State('income', 'value'),
    State('savings_rate', 'value'),
    State('years', 'value'),
    State('init_savings', 'value'),
    State('monthly_withdrawal', 'value'),
    State('strategy', 'value')
)
def update_plan(n_clicks, income, savings_rate, years, init_savings, monthly_withdrawal, strategy):
    if n_clicks == 0:
        raise PreventUpdate

    projected, lower, upper, stock_weight, bond_weight = simulate_retirement_plan(
        income, savings_rate/100, years, init_savings, strategy, monthly_withdrawal=monthly_withdrawal
    )

    total_years = years + 20
    x_vals = list(range(1, total_years + 1))

    fig = go.Figure()

    # Allocation areas
    fig.add_trace(go.Scatter(
        x=x_vals, y=stock_weight,
        name='Equity Allocation (%)',
        mode='none',
        stackgroup='one',
        yaxis='y2',
        fillcolor='rgba(0, 123, 255, 0.5)'
    ))

    fig.add_trace(go.Scatter(
        x=x_vals, y=bond_weight,
        name='Bond Allocation (%)',
        mode='none',
        stackgroup='one',
        yaxis='y2',
        fillcolor='rgba(220, 53, 69, 0.5)'
    ))

    # Confidence interval
    fig.add_trace(go.Scatter(
        x=x_vals, y=upper,
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=lower,
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 200, 255, 0.2)',
        line=dict(width=0),
        name='10-90% Range',
        hoverinfo='skip'
    ))

    # Median portfolio projection
    fig.add_trace(go.Scatter(
        x=x_vals, y=projected,
        mode='lines+markers',
        name='Median Portfolio Value',
        line=dict(color='white', width=3),
        marker=dict(size=4, color='white'),
        yaxis='y1'
    ))

    fig.update_layout(
        title={
            'text': f"Dynamic Glidepath Projection ({'Simulated' if strategy == 'simulated' else 'Standard'})",
            'font': {'color': 'white', 'size': 22},
            'x': 0.5
        },
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#1E1E1E',
        font=dict(color='white', family='Open Sans'),
        xaxis=dict(title='Years', gridcolor='#333', zerolinecolor='#333'),
        yaxis=dict(title='Portfolio Value ($)', gridcolor='#333', zerolinecolor='#333'),
        yaxis2=dict(title='Allocation %', overlaying='y', side='right', range=[0, 1], tickformat='.0%', gridcolor='#333'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )

    summary = [
        html.P(f"Median Value at Retirement: ${projected[years-1]:,.2f}"),
        html.P(f"Median Value After 20 Years: ${projected[-1]:,.2f}")
    ]

    return fig, summary

# Toggle chatbot window visibility
@app.callback(
    Output('chatbot-modal', 'style'),
    [Input('open-chatbot', 'n_clicks'),
     Input('close-chatbot', 'n_clicks')],
    State('chatbot-modal', 'style')
)
def toggle_chatbot(open_clicks, close_clicks, style):
    if style is None:
        style = {'display': 'none'}

    ctx = dash.callback_context
    if not ctx.triggered:
        raise PreventUpdate

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'open-chatbot':
        style['display'] = 'block'
    elif button_id == 'close-chatbot':
        style['display'] = 'none'

    return style

@app.callback(
    Output('chatbot-response', 'children'),
    Input('ask-button', 'n_clicks'),
    State('user-question', 'value')
)
def update_chatbot(n_clicks, user_input):
    global chat_history
    if not n_clicks or not user_input:
        raise PreventUpdate

    chat_history.append(f"User: {user_input}")

    typing_placeholder = html.Div("Assistant is typing . . .", style={
        'fontStyle': 'italic',
        'color': 'gray',
        'padding': '10px',
        'textAlign': 'left',
        'animation': 'typing 1.5s infinite'
    })

    time.sleep(0.5)

    full_prompt = f"{latest_simulation_summary}\n\n{chr(10).join(chat_history)}\n\nAssistant:"
    response = query_ollama(full_prompt)
    chat_history.append(f"Assistant: {response}")

    chat_bubbles = []
    for turn in chat_history:
        role, text = turn.split(":", 1)
        align = 'left' if role.strip() == 'Assistant' else 'right'
        color = '#f1f1f1' if role.strip() == 'Assistant' else '#0074D9'
        text_color = 'black' if role.strip() == 'Assistant' else 'white'
        chat_bubbles.append(dcc.Markdown(text.strip(), style={
            'backgroundColor': color,
            'color': text_color,
            'padding': '8px 12px',
            'borderRadius': '16px',
            'margin': '8px',
            'textAlign': align,
            'maxWidth': '80%',
            'marginLeft': 'auto' if align == 'right' else 'initial',
            'marginRight': 'auto' if align == 'left' else 'initial',
            'whiteSpace': 'pre-wrap',
            'transition': 'box-shadow 0.3s ease-in-out',
            'boxShadow': '0 1px 3px rgba(0,0,0,0.1)',
            'overflowWrap': 'break-word'
        }))

    chat_bubbles.append(html.Div(id='scroll-target'))

    return chat_bubbles

if __name__ == '__main__':
    app.run(debug=True)