import numpy as np
from scipy.stats import norm,multivariate_normal


def black_scholes(S0, K, r, T, sigma, flag="c"):    
    # Calculate d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if flag == "c":  # Call option
        price = (S0 * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif flag == "p":  # Put option
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S0 * norm.cdf(-d1))
    else:
        raise ValueError("Invalid flag, use 'c' for call and 'p' for put")
    
    return price

def general_black_scholes(S0, K, T,r,b, sigma,flag="c"):
    d1 = (np.log(S0 / K) + (b + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if flag == "c":  # Call option
        price = (S0 * np.exp((b-r) * T) * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))
    elif flag == "p":  # Put option
        price = (K * np.exp(-r * T) * norm.cdf(-d2)) - (S0 * np.exp((b-r)*T) * norm.cdf(-d1))
    else:
        raise ValueError("Invalid flag, use 'c' for call and 'p' for put")
    
    return price

def monte_carlo_price(S0, K, r, T, sigma,flag = "c", num_simulations=10000, num_steps=252):
    dt = T / num_steps
    prices = np.zeros((num_simulations, num_steps + 1))
    prices[:, 0] = S0
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        prices[:, t] = prices[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    if flag == "c":  # Call option
        payoffs = np.maximum(prices[:, -1] - K, 0)
    elif flag == "p":  # Put option
        payoffs = np.maximum(K - prices[:, -1], 0)
    else:
        raise ValueError("Invalid flag, use 'c' for call and 'p' for put")    
    price = np.exp(-r * T) * np.mean(payoffs)
    return price

def black76(S0, K, r, T, sigma, flag="c"):    
    # Calculate d1 and d2
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if flag == "c":  # Call option
        price = np.exp(-r * T)*(S0 * norm.cdf(d1) - K * norm.cdf(d2))
    elif flag == "p":  # Put option
        price = np.exp(-r * T)*(K * norm.cdf(-d2) - S0 * norm.cdf(-d1))
    else:
        raise ValueError("Invalid flag, use 'c' for call and 'p' for put")
    
    return price




