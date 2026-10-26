import numpy as np

def calculate_portfolio_return(return_a, weight_a, return_b, weight_b):
    contribution_a = return_a * weight_a
    contribution_b = return_b * weight_b
    portfolio_return = contribution_a + contribution_b

    return portfolio_return

def calculate_portfolio_return_general(returns, weights):
    if len(returns) != len(weights):
        raise ValueError("Returns and weights must be the same length.")

    portfolio_return = 0.0
    for i in range(len(returns)):
        contribution = returns[i] * weights[i]
        portfolio_return += contribution
        
    return portfolio_return

def calculate_portfolio_return_numpy(returns, weights):
    if len(returns) != len(weights):
        raise ValueError("Returns and weights must be the same length.")

    returns_array = np.array(returns)
    weights_array = np.array(weights)

    portfolio_return = np.sum(returns_array * weights_array)

    return portfolio_return

def calculate_volatility(returns):
    returns_array = np.array(returns)

    portfolio_volatility = np.std(returns_array)

    return portfolio_volatility

def calculate_correlation(returns_a, returns_b):
    if len(returns_a) != len(returns_b):
        raise ValueError("Returns must be the same length.")

    returns_a_array = np.array(returns_a)
    returns_b_array = np.array(returns_b)

    correlation_matrix = np.corrcoef(returns_a_array, returns_b_array)
    correlation = correlation_matrix[0, 1]

    return correlation