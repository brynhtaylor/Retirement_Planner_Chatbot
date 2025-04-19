## Application Overview
This is an AI-powered retirement simulation app. It uses glidepath strategies to adjust portfolio allocations from equity to bonds over time.

## Strategy Types
1. **Industry Standard**: Glidepath follows a sigmoid curve from 90% to 10% equity over the investment horizon.
2. **Simulated Optimized**: Uses historical return/covariance data and constraints to create a customized glidepath.

## Simulation Engine
- Monte Carlo simulation with 1000 iterations
- Returns sampled from multivariate normal distributions
- Withdrawals and contributions handled annually

## Key Inputs
- Net income, savings rate, years until retirement
- Monthly withdrawals in retirement
- Glidepath strategy selection

## Outputs
- Median projected wealth
- 10–90% confidence intervals
- Asset allocation over time