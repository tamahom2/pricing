import requests
from bs4 import BeautifulSoup

def gold_price():
    # URL of the TradingView page
    url = 'https://www.tradingview.com/symbols/COMEX-GC1!/'

    # Send a GET request to the page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    print(soup)
    # Find the div with the specific class and data-symbol attribute
    price_div = soup.find('div', {'data-symbol': 'COMEX:GC1!'})

    # Extract the last price and additional value
    last_price_span = price_div.find('span', {'class': 'last-JWoJqCpY'})
    print(last_price_span)
    last_price_main = last_price_span.contents[0]  # Main price part
    last_price_fraction = last_price_span.find('span').text  # Fractional part

    # Combine the main price part and fractional part
    current_price = float(f"{last_price_main}{last_price_fraction}")

    return current_price