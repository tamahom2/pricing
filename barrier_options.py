import numpy as np
from scipy.stats import norm,multivariate_normal
from pricing import general_black_scholes

def discrete_adjusted_barrier(S,H,v,dt):
    """Adjust the barrier following dt
        S:  Asset price
        H:  Barrier
        v: Volatility
        dt: step
    """
    if(H>S):
        return H*np.exp(0.5826*v*np.sqrt(dt))
    elif(H<S):
        return H*np.exp(-0.5826*v*np.sqrt(dt))

def standard_barrier(TypeFlag, S, X, H, K, T, r, b, v):
    """Price a standard barrier option
        TypeFlag: type of the option
        S:  Asset price
        X:  Strike price
        H:  Barrier
        K: Cash rebate 
        T: Time to maturity
        r: Risk-free rate
        b: Cost of carry
        v: Volatility
    """
    if(TypeFlag not in ["cdi","cdo","cui","cuo","pdi","pdo","pui","puo"]):
        raise ValueError("The type of the option should be a Call or Put (Down and out, Down and in, Up and out, Up and In)")
    mu = (b - v ** 2 / 2) / v ** 2
    lambda_ = np.sqrt(mu ** 2 + 2 * r / v ** 2)
    X1 = np.log(S / X) / (v * np.sqrt(T)) + (1 + mu) * v * np.sqrt(T)
    X2 = np.log(S / H) / (v * np.sqrt(T)) + (1 + mu) * v * np.sqrt(T)
    y1 = np.log(H ** 2 / (S * X)) / (v * np.sqrt(T)) + (1 + mu) * v * np.sqrt(T)
    y2 = np.log(H / S) / (v * np.sqrt(T)) + (1 + mu) * v * np.sqrt(T)
    Z = np.log(H / S) / (v * np.sqrt(T)) + lambda_ * v * np.sqrt(T)
    
    if TypeFlag in ["cdi", "cdo"]:
        eta = 1
        phi = 1
    elif TypeFlag in ["cui", "cuo"]:
        eta = -1
        phi = 1
    elif TypeFlag in ["pdi", "pdo"]:
        eta = 1
        phi = -1
    elif TypeFlag in ["pui", "puo"]:
        eta = -1
        phi = -1
    
    f1 = phi * S * np.exp((b - r) * T) * norm.cdf(phi * X1) - phi * X * np.exp(-r * T) * norm.cdf(phi * X1 - phi * v * np.sqrt(T))
    f2 = phi * S * np.exp((b - r) * T) * norm.cdf(phi * X2) - phi * X * np.exp(-r * T) * norm.cdf(phi * X2 - phi * v * np.sqrt(T))
    f3 = phi * S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y1) - phi * X * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y1 - eta * v * np.sqrt(T))
    f4 = phi * S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y2) - phi * X * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y2 - eta * v * np.sqrt(T))
    f5 = K * np.exp(-r * T) * (norm.cdf(eta * X2 - eta * v * np.sqrt(T)) - (H / S) ** (2 * mu) * norm.cdf(eta * y2 - eta * v * np.sqrt(T)))
    f6 = K * ((H / S) ** (mu + lambda_) * norm.cdf(eta * Z) + (H / S) ** (mu - lambda_) * norm.cdf(eta * Z - 2 * eta * lambda_ * v * np.sqrt(T)))
    print(f1)
    print(f2)
    print(f3)
    print(f4)
    print(f5)
    print(f6)
    if X > H:
        if TypeFlag == "cdi":
            return f3 + f5
        elif TypeFlag == "cui":
            return f1 + f5
        elif TypeFlag == "pdi":
            return f2 - f3 + f4 + f5
        elif TypeFlag == "pui":
            return f1 - f2 + f4 + f5
        elif TypeFlag == "cdo":
            return f1 - f3 + f6
        elif TypeFlag == "cuo":
            return f6
        elif TypeFlag == "pdo":
            return f1 - f2 + f3 - f4 + f6
        elif TypeFlag == "puo":
            return f2 - f4 + f6
    elif X < H:
        if TypeFlag == "cdi":
            return f1 - f2 + f4 + f5
        elif TypeFlag == "cui":
            return f2 - f3 + f4 + f5
        elif TypeFlag == "pdi":
            return f1 + f5
        elif TypeFlag == "pui":
            return f3 + f5
        elif TypeFlag == "cdo":
            return f2 + f6 - f4
        elif TypeFlag == "cuo":
            return f1 - f2 + f3 - f4 + f6
        elif TypeFlag == "pdo":
            return f6
        elif TypeFlag == "puo":
            return f1 - f3 + f6

def monte_carlo_barrier(TypeFlag, S, X, H, T, r, v, num_simulations=10000, num_steps=252):
    """Price a standard barrier option using Monte Carlo simulation.
    
    Parameters:
        TypeFlag (str): Type of the option ('cdi', 'cdo', 'cui', 'cuo', 'pdi', 'pdo', 'pui', 'puo')
        S (float): Current asset price
        X (float): Strike price
        H (float): Barrier level
        T (float): Time to maturity
        r (float): Risk-free rate
        v (float): Volatility
        num_simulations (int): Number of Monte Carlo simulations (default: 10000)
        num_steps (int): Number of time steps in each simulation (default: 100)
    
    Returns:
        float: Estimated option price
    """
    if TypeFlag not in ["cdi", "cdo", "cui", "cuo", "pdi", "pdo", "pui", "puo"]:
        raise ValueError("The type of the option should be a Call or Put (Down and out, Down and in, Up and out, Up and In)")
    
    dt = T / num_steps
    discount_factor = np.exp(-r * T)
    S_paths = np.zeros((num_simulations, num_steps + 1))
    S_paths[:, 0] = S
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * v**2) * dt + v * np.sqrt(dt) * Z)
    
    # Check barrier conditions
    if TypeFlag in ["cdo", "pdo","cdi", "pdi"]:
        barrier_crossed = np.any(S_paths <= H, axis=1)
    elif TypeFlag in ["cuo", "puo","cui", "pui"]:
        barrier_crossed = np.any(S_paths >= H, axis=1)
    
    # Calculate payoffs
    if TypeFlag in ["cdo", "cuo", "cdi", "cui"]:
        payoffs = np.maximum(S_paths[:, -1] - X, 0)
    elif TypeFlag in ["pdo", "puo", "pdi", "pui"]:
        payoffs = np.maximum(X - S_paths[:, -1], 0)

    if TypeFlag in ["cdo", "pdo", "cuo", "puo"]:
        payoffs[barrier_crossed] = 0  # Barrier-out options: Set payoff to zero if barrier crossed
    elif TypeFlag in ["cdi", "pdi", "cui", "pui"]:
        payoffs[~barrier_crossed] = 0
    
    discounted_payoffs = discount_factor * payoffs
    
    option_price = np.mean(discounted_payoffs)
    return option_price

def double_barrier(TypeFlag, S, X, L, U, T, r, b, v, delta1, delta2):
    E = L * np.exp(delta1 * T)
    F = U * np.exp(delta1 * T)
    Sum1 = 0
    Sum2 = 0
    
    if TypeFlag in ["co", "ci"]:
        for n in range(-5, 6):
            d1 = (np.log(S * U ** (2 * n) / (X * L ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            d2 = (np.log(S * U ** (2 * n) / (F * L ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            d3 = (np.log(L ** (2 * n + 2) / (X * S * U ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            d4 = (np.log(L ** (2 * n + 2) / (F * S * U ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            mu1 = 2 * (b - delta2 - n * (delta1 - delta2)) / v ** 2 + 1
            mu2 = 2 * n * (delta1 - delta2) / v ** 2
            mu3 = 2 * (b - delta2 + n * (delta1 - delta2)) / v ** 2 + 1
            Sum1 += (U ** n / L ** n) ** mu1 * (L / S) ** mu2 * (norm.cdf(d1) - norm.cdf(d2)) - (L ** (n + 1) / (U ** n * S)) ** mu3 * (norm.cdf(d3) - norm.cdf(d4))
            Sum2 += (U ** n / L ** n) ** (mu1 - 2) * (L / S) ** mu2 * (norm.cdf(d1 - v * np.sqrt(T)) - norm.cdf(d2 - v * np.sqrt(T))) - (L ** (n + 1) / (U ** n * S)) ** (mu3 - 2) * (norm.cdf(d3 - v * np.sqrt(T)) - norm.cdf(d4 - v * np.sqrt(T)))
        OutValue = S * np.exp((b - r) * T) * Sum1 - X * np.exp(-r * T) * Sum2
        
    elif TypeFlag in ["po", "pi"]:
        for n in range(-5, 6):
            d1 = (np.log(S * U ** (2 * n) / (E * L ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            d2 = (np.log(S * U ** (2 * n) / (X * L ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            d3 = (np.log(L ** (2 * n + 2) / (E * S * U ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            d4 = (np.log(L ** (2 * n + 2) / (X * S * U ** (2 * n))) + (b + v ** 2 / 2) * T) / (v * np.sqrt(T))
            mu1 = 2 * (b - delta2 - n * (delta1 - delta2)) / v ** 2 + 1
            mu2 = 2 * n * (delta1 - delta2) / v ** 2
            mu3 = 2 * (b - delta2 + n * (delta1 - delta2)) / v ** 2 + 1
            Sum1 += (U ** n / L ** n) ** mu1 * (L / S) ** mu2 * (norm.cdf(d1) - norm.cdf(d2)) - (L ** (n + 1) / (U ** n * S)) ** mu3 * (norm.cdf(d3) - norm.cdf(d4))
            Sum2 += (U ** n / L ** n) ** (mu1 - 2) * (L / S) ** mu2 * (norm.cdf(d1 - v * np.sqrt(T)) - norm.cdf(d2 - v * np.sqrt(T))) - (L ** (n + 1) / (U ** n * S)) ** (mu3 - 2) * (norm.cdf(d3 - v * np.sqrt(T)) - norm.cdf(d4 - v * np.sqrt(T)))
        OutValue = X * np.exp(-r * T) * Sum2 - S * np.exp((b - r) * T) * Sum1

    if TypeFlag in ["co", "po"]:
        return OutValue
    elif TypeFlag == "ci":
        return general_black_scholes(S, X, T, r, b, v,"c") - OutValue
    elif TypeFlag == "pi":
        return general_black_scholes(S, X, T, r, b, v,"p") - OutValue

def monte_carlo_double_barrier(TypeFlag, S, X, L, U, T, mean, v, num_simulations=10000, num_steps=100):
    """Price a double barrier option using Monte Carlo simulation.
    
    Parameters:
        TypeFlag (str): Type of the option ('cko','cki','pko','pki')
        S (float): Current asset price
        X (float): Strike price
        L (float): Lower Barrier level
        U (float): Upper Barrier level
        T (float): Time to maturity
        r (float): Risk-free rate
        v (float): Volatility
        num_simulations (int): Number of Monte Carlo simulations (default: 10000)
        num_steps (int): Number of time steps in each simulation (default: 100)
    
    Returns:
        float: Estimated option price
    """
    if TypeFlag not in ["cko","cki","pko","pki"]:
        raise ValueError("The type of the option should be a Call or Put (Down and out, Down and in, Up and out, Up and In)")
    
    S_paths = np.zeros((num_simulations, num_steps + 1))
    S_paths[:, 0] = S
    for t in range(1, num_steps + 1):
        Z = norm.ppf(np.random.rand(num_simulations), loc = mean, scale = v)
        S_paths[:, t] = S_paths[:, t-1] * np.exp(Z)
    
    barriers_crossed = np.any((S_paths <= L) | (S_paths >= U), axis=1)
    # Calculate payoffs
    if TypeFlag in ["cki","cko"]:
        payoffs = np.maximum(S_paths[:, -1] - X, 0)
    elif TypeFlag in ["pki","pko"]:
        payoffs = np.maximum(X - S_paths[:, -1], 0)

    if TypeFlag in ["cki","pki"]:
        payoffs[~barriers_crossed] = 0  # Barrier-in options: Set payoff to zero if barrier not crossed
    elif TypeFlag in ["cko","pko"]:
        payoffs[barriers_crossed] = 0
    discounted_payoffs = payoffs
    
    option_price = np.mean(discounted_payoffs)
    return option_price

def CBND(a, b, rho):
    return multivariate_normal.cdf([a, b], mean=[0, 0], cov=[[1, rho], [rho, 1]])

def partial_time_barrier(TypeFlag, S, X, H, t1, T2, r, b, v):
    d1 = (np.log(S / X) + (b + v ** 2 / 2) * T2) / (v * np.sqrt(T2))
    d2 = d1 - v * np.sqrt(T2)
    f1 = (np.log(S / X) + 2 * np.log(H / S) + (b + v ** 2 / 2) * T2) / (v * np.sqrt(T2))
    f2 = f1 - v * np.sqrt(T2)
    e1 = (np.log(S / H) + (b + v ** 2 / 2) * t1) / (v * np.sqrt(t1))
    e2 = e1 - v * np.sqrt(t1)
    e3 = e1 + 2 * np.log(H / S) / (v * np.sqrt(t1))
    e4 = e3 - v * np.sqrt(t1)
    mu = (b - v ** 2 / 2) / v ** 2
    rho = np.sqrt(t1 / T2)
    g1 = (np.log(S / H) + (b + v ** 2 / 2) * T2) / (v * np.sqrt(T2))
    g2 = g1 - v * np.sqrt(T2)
    g3 = g1 + 2 * np.log(H / S) / (v * np.sqrt(T2))
    g4 = g3 - v * np.sqrt(T2)
    
    z1 = norm.cdf(e2) - (H / S) ** (2 * mu) * norm.cdf(e4)
    z2 = norm.cdf(-e2) - (H / S) ** (2 * mu) * norm.cdf(-e4)
    z3 = CBND(g2, e2, rho) - (H / S) ** (2 * mu) * CBND(g4, -e4, -rho)
    z4 = CBND(-g2, -e2, rho) - (H / S) ** (2 * mu) * CBND(-g4, e4, -rho)
    z5 = norm.cdf(e1) - (H / S) ** (2 * (mu + 1)) * norm.cdf(e3)
    z6 = norm.cdf(-e1) - (H / S) ** (2 * (mu + 1)) * norm.cdf(-e3)
    z7 = CBND(g1, e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(g3, -e3, -rho)
    z8 = CBND(-g1, -e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(-g3, e3, -rho)
    
    if TypeFlag == "cdoA":
        eta = 1
    elif TypeFlag == "cuoA":
        eta = -1

    if TypeFlag in ["cdoA", "cuoA"]:  # call down-and-out and up-and-out type A
        partial_time_barrier_price = (
            S * np.exp((b - r) * T2) * (CBND(d1, eta * e1, eta * rho) - (H / S) ** (2 * (mu + 1)) * CBND(f1, eta * e3, eta * rho)) -
            X * np.exp(-r * T2) * (CBND(d2, eta * e2, eta * rho) - (H / S) ** (2 * mu) * CBND(f2, eta * e4, eta * rho))
        )
    elif TypeFlag == "cdoB2" and X < H:  # call down-and-out type B2
        partial_time_barrier_price = (
            S * np.exp((b - r) * T2) * (CBND(g1, e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(g3, -e3, -rho)) -
            X * np.exp(-r * T2) * (CBND(g2, e2, rho) - (H / S) ** (2 * mu) * CBND(g4, -e4, -rho))
        )
    elif TypeFlag == "cdoB2" and X > H:
        partial_time_barrier_price = partial_time_barrier("coB1", S, X, H, t1, T2, r, b, v)
    elif TypeFlag == "cuoB2" and X < H:  # call up-and-out type B2
        partial_time_barrier_price = (
            S * np.exp((b - r) * T2) * (CBND(-g1, -e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(-g3, e3, -rho)) -
            X * np.exp(-r * T2) * (CBND(-g2, -e2, rho) - (H / S) ** (2 * mu) * CBND(-g4, e4, -rho)) -
            S * np.exp((b - r) * T2) * (CBND(-d1, -e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(e3, -f1, -rho)) +
            X * np.exp(-r * T2) * (CBND(-d2, -e2, rho) - (H / S) ** (2 * mu) * CBND(e4, -f2, -rho))
        )
    elif TypeFlag == "coB1" and X > H:  # call out type B1
        partial_time_barrier_price = (
            S * np.exp((b - r) * T2) * (CBND(d1, e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(f1, -e3, -rho)) -
            X * np.exp(-r * T2) * (CBND(d2, e2, rho) - (H / S) ** (2 * mu) * CBND(f2, -e4, -rho))
        )
    elif TypeFlag == "coB1" and X < H:
        partial_time_barrier_price = (
            S * np.exp((b - r) * T2) * (CBND(-g1, -e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(-g3, e3, -rho)) -
            X * np.exp(-r * T2) * (CBND(-g2, -e2, rho) - (H / S) ** (2 * mu) * CBND(-g4, e4, -rho)) -
            S * np.exp((b - r) * T2) * (CBND(-d1, -e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(-f1, e3, -rho)) +
            X * np.exp(-r * T2) * (CBND(-d2, -e2, rho) - (H / S) ** (2 * mu) * CBND(-f2, e4, -rho)) +
            S * np.exp((b - r) * T2) * (CBND(g1, e1, rho) - (H / S) ** (2 * (mu + 1)) * CBND(g3, -e3, -rho)) -
            X * np.exp(-r * T2) * (CBND(g2, e2, rho) - (H / S) ** (2 * mu) * CBND(g4, -e4, -rho))
        )
    elif TypeFlag == "pdoA":  # put down-and-out and up-and-out type A
        partial_time_barrier_price = (
            partial_time_barrier("cdoA", S, X, H, t1, T2, r, b, v) -
            S * np.exp((b - r) * T2) * z5 + X * np.exp(-r * T2) * z1
        )
    elif TypeFlag == "puoA":
        partial_time_barrier_price = (
            partial_time_barrier("cuoA", S, X, H, t1, T2, r, b, v) -
            S * np.exp((b - r) * T2) * z6 + X * np.exp(-r * T2) * z2
        )
    elif TypeFlag == "poB1":  # put out type B1
        partial_time_barrier_price = (
            partial_time_barrier("coB1", S, X, H, t1, T2, r, b, v) -
            S * np.exp((b - r) * T2) * z8 + X * np.exp(-r * T2) * z4 -
            S * np.exp((b - r) * T2) * z7 + X * np.exp(-r * T2) * z3
        )
    elif TypeFlag == "pdoB2":  # put down-and-out type B2
        partial_time_barrier_price = (
            partial_time_barrier("cdoB2", S, X, H, t1, T2, r, b, v) -
            S * np.exp((b - r) * T2) * z7 + X * np.exp(-r * T2) * z3
        )
    elif TypeFlag == "puoB2":  # put up-and-out type B2
        partial_time_barrier_price = (
            partial_time_barrier("cuoB2", S, X, H, t1, T2, r, b, v) -
            S * np.exp((b - r) * T2) * z8 + X * np.exp(-r * T2) * z4
        )
    return partial_time_barrier_price

def two_asset_barrier(TypeFlag, S1, S2, X, H, T, r, b1, b2, v1, v2, rho):
    mu1 = b1 - v1 ** 2 / 2
    mu2 = b2 - v2 ** 2 / 2

    d1 = (np.log(S1 / X) + (mu1 + v1 ** 2 / 2) * T) / (v1 * np.sqrt(T))
    d2 = d1 - v1 * np.sqrt(T)
    d3 = d1 + 2 * rho * np.log(H / S2) / (v2 * np.sqrt(T))
    d4 = d2 + 2 * rho * np.log(H / S2) / (v2 * np.sqrt(T))
    e1 = (np.log(H / S2) - (mu2 + rho * v1 * v2) * T) / (v2 * np.sqrt(T))
    e2 = e1 + rho * v1 * np.sqrt(T)
    e3 = e1 - 2 * np.log(H / S2) / (v2 * np.sqrt(T))
    e4 = e2 - 2 * np.log(H / S2) / (v2 * np.sqrt(T))
    
    if TypeFlag in ["cuo", "cui"]:
        eta, phi = 1, 1
    elif TypeFlag in ["cdo", "cdi"]:
        eta, phi = 1, -1
    elif TypeFlag in ["puo", "pui"]:
        eta, phi = -1, 1
    elif TypeFlag in ["pdo", "pdi"]:
        eta, phi = -1, -1
    else:
        raise ValueError("Invalid TypeFlag value")
    
    KnockOutValue = (eta * S1 * np.exp((b1 - r) * T) * 
                     (CBND(eta * d1, phi * e1, -eta * phi * rho) - 
                      np.exp(2 * (mu2 + rho * v1 * v2) * np.log(H / S2) / v2 ** 2) * 
                      CBND(eta * d3, phi * e3, -eta * phi * rho)) - 
                     eta * np.exp(-r * T) * X * 
                     (CBND(eta * d2, phi * e2, -eta * phi * rho) - 
                      np.exp(2 * mu2 * np.log(H / S2) / v2 ** 2) * 
                      CBND(eta * d4, phi * e4, -eta * phi * rho)))
    
    if TypeFlag in ["cuo", "cdo", "puo", "pdo"]:
        return KnockOutValue
    elif TypeFlag in ["cui", "cdi"]:
        return general_black_scholes(S1, X, T, r, b1, v1, "c") - KnockOutValue
    elif TypeFlag in ["pui", "pdi"]:
        return general_black_scholes(S1, X, T, r, b1, v1, "p") - KnockOutValue
    else:
        raise ValueError("Invalid TypeFlag value")

def partial_time_two_asset_barrier(TypeFlag, S1, S2, X, H, t1, T2, r, b1, b2, v1, v2, rho):
    if TypeFlag in ["cdo", "pdo", "cdi", "pdi"]:
        phi = -1
    else:
        phi = 1
    
    if TypeFlag in ["cdo", "cuo", "cdi", "cui"]:
        eta = 1
    else:
        eta = -1
    
    mu1 = b1 - v1 ** 2 / 2
    mu2 = b2 - v2 ** 2 / 2
    
    d1 = (np.log(S1 / X) + (mu1 + v1 ** 2) * T2) / (v1 * np.sqrt(T2))
    d2 = d1 - v1 * np.sqrt(T2)
    d3 = d1 + 2 * rho * np.log(H / S2) / (v2 * np.sqrt(T2))
    d4 = d2 + 2 * rho * np.log(H / S2) / (v2 * np.sqrt(T2))
    
    e1 = (np.log(H / S2) - (mu2 + rho * v1 * v2) * t1) / (v2 * np.sqrt(t1))
    e2 = e1 + rho * v1 * np.sqrt(t1)
    e3 = e1 - 2 * np.log(H / S2) / (v2 * np.sqrt(t1))
    e4 = e2 - 2 * np.log(H / S2) / (v2 * np.sqrt(t1))
    
    OutBarrierValue = (eta * S1 * np.exp((b1 - r) * T2) * 
                       (CBND(eta * d1, phi * e1, -eta * phi * rho * np.sqrt(t1 / T2)) - 
                        np.exp(2 * np.log(H / S2) * (mu2 + rho * v1 * v2) / (v2 ** 2)) * 
                        CBND(eta * d3, phi * e3, -eta * phi * rho * np.sqrt(t1 / T2))) - 
                       eta * np.exp(-r * T2) * X * 
                       (CBND(eta * d2, phi * e2, -eta * phi * rho * np.sqrt(t1 / T2)) - 
                        np.exp(2 * np.log(H / S2) * mu2 / (v2 ** 2)) * 
                        CBND(eta * d4, phi * e4, -eta * phi * rho * np.sqrt(t1 / T2))))
    
    if TypeFlag in ["cdo", "cuo", "pdo", "puo"]:
        return OutBarrierValue
    elif TypeFlag in ["cui", "cdi"]:
        return general_black_scholes(S1, X, T2, r, b1, v1, "c") - OutBarrierValue
    elif TypeFlag in ["pui", "pdi"]:
        return general_black_scholes(S1, X, T2, r, b1, v1, "p") - OutBarrierValue
    else:
        raise ValueError("Invalid TypeFlag value")

def PartialFixedLB(CallPutFlag, S, X, t1, T2, r, b, v):
    d1 = (np.log(S / X) + (b + v ** 2 / 2) * T2) / (v * np.sqrt(T2))
    d2 = d1 - v * np.sqrt(T2)
    e1 = ((b + v ** 2 / 2) * (T2 - t1)) / (v * np.sqrt(T2 - t1))
    e2 = e1 - v * np.sqrt(T2 - t1)
    f1 = (np.log(S / X) + (b + v ** 2 / 2) * t1) / (v * np.sqrt(t1))
    f2 = f1 - v * np.sqrt(t1)

    if CallPutFlag == "c":
        partial_fixed_lb = (
            S * np.exp((b - r) * T2) * norm.cdf(d1) -
            np.exp(-r * T2) * X * norm.cdf(d2) +
            S * np.exp(-r * T2) * (v ** 2 / (2 * b)) * (
                -(S / X) ** (-2 * b / v ** 2) * CBND(d1 - 2 * b * np.sqrt(T2) / v, -f1 + 2 * b * np.sqrt(t1) / v, -np.sqrt(t1 / T2)) +
                np.exp(b * T2) * CBND(e1, d1, np.sqrt(1 - t1 / T2))
            ) -
            S * np.exp((b - r) * T2) * CBND(-e1, d1, -np.sqrt(1 - t1 / T2)) -
            X * np.exp(-r * T2) * CBND(f2, -d2, -np.sqrt(t1 / T2)) +
            np.exp(-b * (T2 - t1)) * (1 - v ** 2 / (2 * b)) * S * np.exp((b - r) * T2) * norm.cdf(f1) * norm.cdf(-e2)
        )
    elif CallPutFlag == "p":
        partial_fixed_lb = (
            X * np.exp(-r * T2) * norm.cdf(-d2) -
            S * np.exp((b - r) * T2) * norm.cdf(-d1) +
            S * np.exp(-r * T2) * (v ** 2 / (2 * b)) * (
                (S / X) ** (-2 * b / v ** 2) * CBND(-d1 + 2 * b * np.sqrt(T2) / v, f1 - 2 * b * np.sqrt(t1) / v, -np.sqrt(t1 / T2)) -
                np.exp(b * T2) * CBND(-e1, -d1, np.sqrt(1 - t1 / T2))
            ) +
            S * np.exp((b - r) * T2) * CBND(e1, -d1, -np.sqrt(1 - t1 / T2)) +
            X * np.exp(-r * T2) * CBND(-f2, d2, -np.sqrt(t1 / T2)) -
            np.exp(-b * (T2 - t1)) * (1 - v ** 2 / (2 * b)) * S * np.exp((b - r) * T2) * norm.cdf(-f1) * norm.cdf(e2)
        )
    else:
        raise ValueError("Invalid CallPutFlag value")

    return partial_fixed_lb


def look_barrier(TypeFlag, S, X, H, t1, T2, r, b, v):
    hh = np.log(H / S)
    K = np.log(X / S)
    mu1 = b - v ** 2 / 2
    mu2 = b + v ** 2 / 2
    rho = np.sqrt(t1 / T2)

    if TypeFlag in ["cuo", "cui"]:
        eta = 1
        m = min(hh, K)
    elif TypeFlag in ["pdo", "pdi"]:
        eta = -1
        m = max(hh, K)
    else:
        raise ValueError("Invalid TypeFlag value")

    g1 = (norm.cdf(eta * (hh - mu2 * t1) / (v * np.sqrt(t1))) - np.exp(2 * mu2 * hh / v ** 2) * norm.cdf(eta * (-hh - mu2 * t1) / (v * np.sqrt(t1)))) - (norm.cdf(eta * (m - mu2 * t1) / (v * np.sqrt(t1))) - np.exp(2 * mu2 * hh / v ** 2) * norm.cdf(eta * (m - 2 * hh - mu2 * t1) / (v * np.sqrt(t1))))
    g2 = (norm.cdf(eta * (hh - mu1 * t1) / (v * np.sqrt(t1))) - np.exp(2 * mu1 * hh / v ** 2) * norm.cdf(eta * (-hh - mu1 * t1) / (v * np.sqrt(t1)))) - (norm.cdf(eta * (m - mu1 * t1) / (v * np.sqrt(t1))) - np.exp(2 * mu1 * hh / v ** 2) * norm.cdf(eta * (m - 2 * hh - mu1 * t1) / (v * np.sqrt(t1))))

    part1 = S * np.exp((b - r) * T2) * (1 + v ** 2 / (2 * b)) * (CBND(eta * (m - mu2 * t1) / (v * np.sqrt(t1)), eta * (-K + mu2 * T2) / (v * np.sqrt(T2)), -rho) - np.exp(2 * mu2 * hh / v ** 2) * CBND(eta * (m - 2 * hh - mu2 * t1) / (v * np.sqrt(t1)), eta * (2 * hh - K + mu2 * T2) / (v * np.sqrt(T2)), -rho))
    part2 = -np.exp(-r * T2) * X * (CBND(eta * (m - mu1 * t1) / (v * np.sqrt(t1)), eta * (-K + mu1 * T2) / (v * np.sqrt(T2)), -rho) - np.exp(2 * mu1 * hh / v ** 2) * CBND(eta * (m - 2 * hh - mu1 * t1) / (v * np.sqrt(t1)), eta * (2 * hh - K + mu1 * T2) / (v * np.sqrt(T2)), -rho))
    part3 = -np.exp(-r * T2) * v ** 2 / (2 * b) * (S * (S / X) ** (-2 * b / v ** 2) * CBND(eta * (m + mu1 * t1) / (v * np.sqrt(t1)), eta * (-K - mu1 * T2) / (v * np.sqrt(T2)), -rho) - H * (H / X) ** (-2 * b / v ** 2) * CBND(eta * (m - 2 * hh + mu1 * t1) / (v * np.sqrt(t1)), eta * (2 * hh - K - mu1 * T2) / (v * np.sqrt(T2)), -rho))
    part4 = S * np.exp((b - r) * T2) * ((1 + v ** 2 / (2 * b)) * norm.cdf(eta * mu2 * (T2 - t1) / (v * np.sqrt(T2 - t1))) + np.exp(-b * (T2 - t1)) * (1 - v ** 2 / (2 * b)) * norm.cdf(eta * (-mu1 * (T2 - t1)) / (v * np.sqrt(T2 - t1)))) * g1 - np.exp(-r * T2) * X * g2
    OutValue = eta * (part1 + part2 + part3 + part4)

    if TypeFlag in ["cuo", "pdo"]:
        return OutValue
    elif TypeFlag == "cui":
        return PartialFixedLB("c", S, X, t1, T2, r, b, v) - OutValue
    elif TypeFlag == "pdi":
        return PartialFixedLB("p", S, X, t1, T2, r, b, v) - OutValue
    else:
        raise ValueError("Invalid TypeFlag value")


def SoftBarrier(TypeFlag, S, X, L, U, T, r, b, v):
    if TypeFlag in ["cdi", "cdo"]:
        eta = 1
    else:
        eta = -1

    mu = (b + v ** 2 / 2) / v ** 2
    lambda1 = np.exp(-0.5 * v ** 2 * T * (mu + 0.5) * (mu - 0.5))
    lambda2 = np.exp(-0.5 * v ** 2 * T * (mu - 0.5) * (mu - 1.5))
    d1 = np.log(U ** 2 / (S * X)) / (v * np.sqrt(T)) + mu * v * np.sqrt(T)
    d2 = d1 - (mu + 0.5) * v * np.sqrt(T)
    d3 = np.log(U ** 2 / (S * X)) / (v * np.sqrt(T)) + (mu - 1) * v * np.sqrt(T)
    d4 = d3 - (mu - 0.5) * v * np.sqrt(T)
    e1 = np.log(L ** 2 / (S * X)) / (v * np.sqrt(T)) + mu * v * np.sqrt(T)
    e2 = e1 - (mu + 0.5) * v * np.sqrt(T)
    e3 = np.log(L ** 2 / (S * X)) / (v * np.sqrt(T)) + (mu - 1) * v * np.sqrt(T)
    e4 = e3 - (mu - 0.5) * v * np.sqrt(T)

    Value = (eta / (U - L) *
             (S * np.exp((b - r) * T) * S ** (-2 * mu) *
              (S * X) ** (mu + 0.5) / (2 * (mu + 0.5)) *
              ((U ** 2 / (S * X)) ** (mu + 0.5) * norm.cdf(eta * d1) -
               lambda1 * norm.cdf(eta * d2) -
               (L ** 2 / (S * X)) ** (mu + 0.5) * norm.cdf(eta * e1) +
               lambda1 * norm.cdf(eta * e2)) -
              X * np.exp(-r * T) * S ** (-2 * (mu - 1)) *
              (S * X) ** (mu - 0.5) / (2 * (mu - 0.5)) *
              ((U ** 2 / (S * X)) ** (mu - 0.5) * norm.cdf(eta * d3) -
               lambda2 * norm.cdf(eta * d4) -
               (L ** 2 / (S * X)) ** (mu - 0.5) * norm.cdf(eta * e3) +
               lambda2 * norm.cdf(eta * e4))))

    if TypeFlag in ["cdi", "pui"]:
        return Value
    elif TypeFlag == "cdo":
        return general_black_scholes(S, X, T, r, b, v, "c") - Value
    elif TypeFlag == "puo":
        return general_black_scholes(S, X, T, r, b, v, "p") - Value
    else:
        raise ValueError("Invalid TypeFlag value")

def BinaryBarrier(TypeFlag, S, X, H, K, T, r, b, v, eta, phi):
    mu = (b - v ** 2 / 2) / v ** 2
    lambda_ = np.sqrt(mu ** 2 + 2 * r / v ** 2)
    X1 = np.log(S / X) / (v * np.sqrt(T)) + (mu + 1) * v * np.sqrt(T)
    X2 = np.log(S / H) / (v * np.sqrt(T)) + (mu + 1) * v * np.sqrt(T)
    y1 = np.log(H ** 2 / (S * X)) / (v * np.sqrt(T)) + (mu + 1) * v * np.sqrt(T)
    y2 = np.log(H / S) / (v * np.sqrt(T)) + (mu + 1) * v * np.sqrt(T)
    Z = np.log(H / S) / (v * np.sqrt(T)) + lambda_ * v * np.sqrt(T)

    a1 = S * np.exp((b - r) * T) * norm.cdf(phi * X1)
    b1 = K * np.exp(-r * T) * norm.cdf(phi * X1 - phi * v * np.sqrt(T))
    a2 = S * np.exp((b - r) * T) * norm.cdf(phi * X2)
    b2 = K * np.exp(-r * T) * norm.cdf(phi * X2 - phi * v * np.sqrt(T))
    a3 = S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y1)
    b3 = K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y1 - eta * v * np.sqrt(T))
    a4 = S * np.exp((b - r) * T) * (H / S) ** (2 * (mu + 1)) * norm.cdf(eta * y2)
    b4 = K * np.exp(-r * T) * (H / S) ** (2 * mu) * norm.cdf(eta * y2 - eta * v * np.sqrt(T))
    a5 = K * ((H / S) ** (mu + lambda_) * norm.cdf(eta * Z) + (H / S) ** (mu - lambda_) * norm.cdf(eta * Z - 2 * eta * lambda_ * v * np.sqrt(T)))

    if X > H:
        if TypeFlag < 5:
            return a5
        elif TypeFlag < 7:
            return b2 + b4
        elif TypeFlag < 9:
            return a2 + a4
        elif TypeFlag < 11:
            return b2 - b4
        elif TypeFlag < 13:
            return a2 - a4
        elif TypeFlag == 13:
            return b3
        elif TypeFlag == 14:
            return b3
        elif TypeFlag == 15:
            return a3
        elif TypeFlag == 16:
            return a1
        elif TypeFlag == 17:
            return b2 - b3 + b4
        elif TypeFlag == 18:
            return b1 - b2 + b4
        elif TypeFlag == 19:
            return a2 - a3 + a4
        elif TypeFlag == 20:
            return a1 - a2 + a3
        elif TypeFlag == 21:
            return b1 - b3
        elif TypeFlag == 22:
            return 0
        elif TypeFlag == 23:
            return a1 - a3
        elif TypeFlag == 24:
            return 0
        elif TypeFlag == 25:
            return b1 - b2 + b3 - b4
        elif TypeFlag == 26:
            return b2 - b4
        elif TypeFlag == 27:
            return a1 - a2 + a3 - a4
        elif TypeFlag == 28:
            return a2 - a4
    elif X < H:
        if TypeFlag < 5:
            return a5
        elif TypeFlag < 7:
            return b2 + b4
        elif TypeFlag < 9:
            return a2 + a4
        elif TypeFlag < 11:
            return b2 - b4
        elif TypeFlag < 13:
            return a2 - a4
        elif TypeFlag == 13:
            return b1 - b2 + b4
        elif TypeFlag == 14:
            return b2 - b3 + b4
        elif TypeFlag == 15:
            return a1 - a2 + a4
        elif TypeFlag == 16:
            return a2 - a3 + a4
        elif TypeFlag == 17:
            return b1
        elif TypeFlag == 18:
            return b3
        elif TypeFlag == 19:
            return a1
        elif TypeFlag == 20:
            return a3
        elif TypeFlag == 21:
            return b2 - b4
        elif TypeFlag == 22:
            return b1 - b2 + b3 - b4
        elif TypeFlag == 23:
            return a2 - a4
        elif TypeFlag == 24:
            return a1 - a2 + a3 - a4
        elif TypeFlag == 25:
            return 0
        elif TypeFlag == 26:
            return b1 - b3
        elif TypeFlag == 27:
            return 0
        elif TypeFlag == 28:
            return a1 - a3
        

S = 100.0
X = 100.0
H = 115.0
K = 3.0
T = 0.5
r = 0.08
b = 0.04
v = 0.2

print(standard_barrier("cuo",S,X,H,K,T,r,b,v))