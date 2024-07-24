from ticker import future_ticker
import requests
import json
import pandas as pd

def request(ticker,type = "commodity"):
    url = "https://scanner.tradingview.com/options/scan2"
    tickers = future_ticker(ticker) if type=="commodity" else [ticker]
    headers = {
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9,fr-FR;q=0.8,fr;q=0.7,ar;q=0.6",
        "content-type": "text/plain;charset=UTF-8",
        "priority": "u=1, i",
        "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
        "sec-ch-ua-mobile": "?1",
        "sec-ch-ua-platform": "\"Android\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "cookie": "cookiePrivacyPreferenceBannerProduction=notApplicable; _ga=GA1.1.491006717.1719224217; cookiesSettings={\"analytics\":true,\"advertising\":true}; _sp_ses.cf1a=*; _sp_id.cf1a=ebf073c7-a54f-470d-9bbf-f26fa5da69d6.1719224217.6.1719342054.1719314477.a33d3d81-6d5c-4fe2-bffb-59bea6783894; _ga_YVVRYGL0E0=GS1.1.1719342043.7.1.1719342439.56.0.0",
        "Referer": "https://www.tradingview.com/",
        "Referrer-Policy": "origin-when-cross-origin"
    }
    body = {
        "columns": ["ask", "bid", "delta", "expiration", "gamma", "iv", "name", "option-type", "rho", "root", "strike", "theoPrice", "theta", "vega"],
        "filter": [{"left": "type", "operation": "equal", "right": "option"}],
        "ignore_unknown_fields": False,
        "index_filters": [
            {"name": "underlying_symbol", "values": future_ticker(ticker)}
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(body))
    if response.status_code == 200:
        json_data = response.json()
        # Extract the fields and symbols
        fields = json_data['fields']
        symbols = json_data['symbols']

        # Prepare the data for DataFrame
        data = []
        for symbol in symbols:
            symbol_data = symbol['f']
            symbol_data_dict = {fields[i]: symbol_data[i] for i in range(len(fields))}
            symbol_data_dict['symbol'] = symbol['s']
            data.append(symbol_data_dict)

        # Create DataFrame
        df = pd.DataFrame(data)
        return df
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        print(response.text)