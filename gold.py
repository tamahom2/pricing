import pandas as pd
import pandas as pd
import numpy as np
from datetime import datetime
from historical_data import historical_data
from request_option import request
from pricing import black_scholes,monte_carlo_price
from greeks import rho_calc,vega_calc
from risk_free_rate import get_risk_free_rate
import warnings
warnings.simplefilter(action='ignore')

def find_risk_free_rate(actual_price, S0, K, T, sigma, tolerance=1e-6):
    """Compute the risk free rate of a European Option
        actual_price: market observed price
        S0: initial stock price
        K:  strike price
        T:  maturity
        sigma:  implied volatility
        tol: user choosen tolerance
    """
    max_iter = 200 #max number of iterations
    r_old = 0.02 #initial guess

    for k in range(max_iter):
        bs_price = black_scholes(S0, K, r_old, T , sigma)
        Cprime =  rho_calc(r_old,S0, K, T, sigma)
        C = bs_price - actual_price
        r_new = r_old - C/Cprime
        bs_new = black_scholes(S0, K, r_new, T , sigma)
        if (abs(r_old - r_new) < tolerance or abs(bs_new - actual_price) < tolerance):
            break
        r_old = r_new

    r = r_old
    return r

def implied_vol(S0, K, T, r, market_price, flag='c', tol=0.00001):
    """Compute the implied volatility of a European Option
        S0: initial stock price
        K:  strike price
        T:  maturity
        r:  risk-free rate
        market_price: market observed price
        tol: user choosen tolerance
    """
    max_iter = 200 #max number of iterations
    sigma = 0.10 #initial guess

    for k in range(max_iter):
        price = black_scholes(S0, K, r, T, sigma)
        vega =  vega_calc(r, S0, K, T, sigma)*100
        diff = market_price-price
        if (abs(diff) < tol):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma


if __name__ == "__main__":
    tickerYahoo = 'GC=F'
    tickerTV = "COMEX:GC"
    df = request(tickerTV)
    df = df.dropna(subset=['theoPrice'])

    df['expiration'] = pd.to_datetime(df['expiration'], format='%Y%m%d')
    df_sorted = df.sort_values(by=['option-type', 'expiration', 'strike'])
    # Display the sorted DataFrame
    print(df_sorted.head())

    gold_data,S0 = historical_data(tickerYahoo)
    print(gold_data.head())
    print(f"GOLD PRICE IS : {S0}")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Calculate historical volatility (annualized)
    # 252 trading days in a year
    volatility = gold_data['Log Return'].std() * np.sqrt(252)
    print(f"The volatility is {volatility}")
    call_options = df_sorted[df_sorted['option-type'] == 'call']
    call_options = call_options[call_options['strike'] > S0]
    call_options = call_options[call_options['expiration'] > today]
    print(call_options.head())
    call_options['Calculated Price'] = call_options.apply(
        lambda row: black_scholes(S0,row["strike"],0.015,(row["expiration"] - today).days / 365.0,row["iv"]), axis = 1
    )
    call_options['Price Difference'] = call_options['theoPrice'] - call_options['Calculated Price']

    # Display the DataFrame with actual and calculated prices
    print(call_options[['symbol', 'strike', 'bid','ask','theoPrice', 'Calculated Price', 'Price Difference']])
    call_options[['symbol', 'strike', 'theoPrice', 'Calculated Price', 'Price Difference']].to_csv('blackscholespricing.csv', index=False)
    call_options['Monte Carlo Price'] = call_options.apply(
        lambda row: monte_carlo_price(S0,row["strike"],get_risk_free_rate(row["expiration"]),(row["expiration"] - today).days / 365.0,volatility), axis = 1
    )
    call_options['Price Difference'] = call_options['theoPrice'] - call_options['Monte Carlo Price']

    # Display the DataFrame with actual and calculated prices
    print(call_options[['symbol', 'strike', 'theoPrice', 'Monte Carlo Price', 'Price Difference']])
    
    call_options['Risk Free Rate'] = call_options.apply(
        lambda row: find_risk_free_rate(row['theoPrice'],S0,row["strike"],(row["expiration"] - today).days / 365.0,row["iv"]), axis=1
    )
    
    print(call_options[['symbol', 'strike', 'theoPrice', 'Risk Free Rate']])

    call_options['Implied Volatility Calc'] = call_options.apply(
        lambda row: implied_vol(S0,row["strike"],(row["expiration"] - today).days / 365.0,get_risk_free_rate(row["expiration"]),row["theoPrice"]), axis=1
    )
    call_options['Implied Difference'] = call_options['iv'] - call_options['Implied Volatility Calc']
    print("Implied Volatilities")
    print(call_options[['symbol', 'strike', 'iv', 'Implied Volatility Calc',"Implied Difference"]])

    print("The columns of the data frame : ")
    print(call_options.columns)