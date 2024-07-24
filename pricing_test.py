import pandas as pd
import pandas as pd
import numpy as np
from datetime import datetime
from historical_data import historical_data
from request_option import request
from pricing import black_scholes,general_black_scholes,monte_carlo_price,black76
from greeks import rho_calc,vega_calc
from risk_free_rate import get_risk_free_rate
import warnings
from forward_req import forward_request
warnings.simplefilter(action='ignore')

def find_risk_free_rate(actual_price, S0, K, T, sigma, flag = "c", tolerance=1e-6):
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
        bs_price = black_scholes(S0, K, r_old, T , sigma,flag)
        Cprime =  rho_calc(r_old,S0, K, T, sigma,flag)
        C = bs_price - actual_price
        r_new = r_old - C/Cprime
        bs_new = black_scholes(S0, K, r_new, T , sigma,flag)
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
        price = black_scholes(S0, K, r, T, sigma,flag)
        vega =  vega_calc(r, S0, K, T, sigma,flag)
        diff = market_price-price
        if (abs(diff) < tol):
            return sigma
        sigma = sigma + diff/vega # f(x) / f'(x)
    return sigma


def get_price_from_forward_curve(expiration_date, forward_curve):
    """Given an expiration date, get the price from the forward curve."""
    expiration_month_year = expiration_date.strftime("%Y%m")
    for _, option in forward_curve.iterrows():
        option_expiration = option['Expiration']
        option_month_year = option_expiration.strftime("%Y%m")
        if option_month_year == expiration_month_year:
            return option['Close']
    return None

def black_scholes_test(options,option_type,forward_curve,today,flag):
    options[f'Calculated {option_type} Price'] = options.apply(
        lambda row: general_black_scholes(get_price_from_forward_curve(row["expiration"],forward_curve),row["strike"],(row["expiration"] - today).days / 365.0,0.0,0.0,row["iv"],flag), axis = 1
    )

    options["Forward Price"] = options.apply(
        lambda row : get_price_from_forward_curve(row["expiration"],forward_curve), axis = 1
    )

    options['Price Difference'] = options.apply(
        lambda row : 100*(row['theoPrice'] - row[f'Calculated {option_type} Price']) / get_price_from_forward_curve(row["expiration"],forward_curve), axis = 1
    )
    # Display the DataFrame with actual and calculated prices
    new_df = options[options['Price Difference'] > 0.05]
    print(new_df[['symbol', 'strike', 'theoPrice', f'Calculated {option_type} Price', 'Price Difference']].head())
    new_df[['symbol', 'strike','Forward Price', 'theoPrice', f'Calculated {option_type} Price', 'Price Difference']].to_csv("blackscholespricing.csv")

def black76_test(options,option_type,forward_curve,today,flag):
    options[f'Calculated {option_type} Price'] = options.apply(
        lambda row: black76(get_price_from_forward_curve(row["expiration"],forward_curve),row["strike"],get_risk_free_rate(row["expiration"]),(row["expiration"] - today).days / 365.0,row["iv"],flag), axis = 1
    )
    options['Price Difference'] = options['theoPrice'] - options[f'Calculated {option_type} Price']

    # Display the DataFrame with actual and calculated prices
    print(options[['symbol', 'strike', 'theoPrice', f'Calculated {option_type} Price', 'Price Difference']])
    options[['symbol', 'strike', 'theoPrice', f'Calculated {option_type} Price', 'Price Difference']].to_csv("black76pricing.csv")


def monte_carlo_test(options,forward_curve,today,flag):
    options['Monte Carlo Price'] = options.apply(
        lambda row: monte_carlo_price(get_price_from_forward_curve(row["expiration"],forward_curve),row["strike"],get_risk_free_rate(row["expiration"]),(row["expiration"] - today).days / 365.0,row["iv"],flag, 1000), axis = 1
    )
    options['Price Difference'] = options['theoPrice'] - options['Monte Carlo Price']

    # Display the DataFrame with actual and calculated prices
    print(options[['symbol', 'strike', 'theoPrice', 'Monte Carlo Price', 'Price Difference']])


def risk_free_rate_test(options,S0,today,flag):
    options['Risk Free Rate'] = options.apply(
        lambda row: find_risk_free_rate(row['theoPrice'],S0,row["strike"],(row["expiration"] - today).days / 365.0,row["iv"],flag), axis=1
    )
    
    print(options[['symbol', 'strike', 'theoPrice', 'Risk Free Rate']])

def iv_test(options,forward_curve,today,flag):
    options['Implied Volatility Calc'] = options.apply(
        lambda row: implied_vol(get_price_from_forward_curve(row["expiration"],forward_curve),row["strike"],(row["expiration"] - today).days / 365.0,get_risk_free_rate(row["expiration"]),row["theoPrice"],flag), axis=1
    )
    options['Implied Difference'] = options['iv'] - options['Implied Volatility Calc']
    print(options[['symbol', 'strike', 'iv', 'Implied Volatility Calc',"Implied Difference"]])

def pricing_test(tickerYahoo,tickerTV,flag="c",type = "commodity"):
    if(flag!="c" and flag!="p"):
        raise ValueError("Invalid flag, use 'c' for call and 'p' for put")
    option_type = "Call" if flag=="c" else "Put"
    df = request(tickerTV,type)
    forward_curve = forward_request(tickerTV)
    df = df.dropna(subset=['theoPrice'])

    df['expiration'] = pd.to_datetime(df['expiration'], format='%Y%m%d')
    df_sorted = df.sort_values(by=['option-type', 'expiration', 'strike'])

    commo_data,S0 = historical_data(tickerYahoo)
    print(commo_data.head())
    print(f"THE {tickerTV} PRICE IS : {S0}")
    print("We re pricing for "+ option_type +" options")
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Calculate historical volatility (annualized)
    # 252 trading days in a year
    volatility = commo_data['Log Return'].std() * np.sqrt(252)
    print(f"The volatility is {volatility}")
    if(flag == "c"):
        options = df_sorted[df_sorted['option-type'] == 'call']
        options = options[options['strike'] > S0]
    else:
        options = df_sorted[df_sorted['option-type'] == 'put']
        options = options[options['strike'] < S0]
    options = options[options['expiration'] > today]
    print(options.head())
    
    print("BSM test")
    black_scholes_test(options,option_type,forward_curve,today,flag)

    print("Black76 test")
    black76_test(options,option_type,forward_curve,today,flag)

    print("Monte Carlo test")
    monte_carlo_test(options,forward_curve,today,flag)
    
    print("Risk free rate test")
    risk_free_rate_test(options,S0,today,flag)
    
    print("Implied Volatility test")
    iv_test(options,forward_curve,today,flag)

    


if __name__ == "__main__":
    tickerYahoo = "CL=F"
    tickerTV = "NYMEX:CL"
    pricing_test(tickerYahoo,tickerTV,"p","stock")
