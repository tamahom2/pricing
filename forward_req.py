import requests
import json
import pandas as pd
import matplotlib.pyplot as plt



def forward_request(ticker):
    # URL
    url = "https://scanner.tradingview.com/futures/scan"

    # Headers
    headers = {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9,fr-FR;q=0.8,fr;q=0.7,ar;q=0.6",
        "content-type": "text/plain;charset=UTF-8",
        "cookie": "cookiePrivacyPreferenceBannerProduction=notApplicable; _ga=GA1.1.491006717.1719224217; cookiesSettings={\"analytics\":true,\"advertising\":true}; device_t=YmhMc0JBOjA.t-PdoCwm3W_daZjQj33f9-TFVcUcUuSdXopsms5JDP8; sessionid=efvqmju3dwzx206fhdm2vtnbh5j0izd7; sessionid_sign=v2:KRn4pCSmIPOYkeV9mXULSvpnxuzbM1F3pibcfk42zX8=; png=4751eca5-a4a4-4b5a-a1f7-f599f8ca6a2c; etg=4751eca5-a4a4-4b5a-a1f7-f599f8ca6a2c; cachec=4751eca5-a4a4-4b5a-a1f7-f599f8ca6a2c; tv_ecuid=4751eca5-a4a4-4b5a-a1f7-f599f8ca6a2c; _sp_ses.cf1a=*; _sp_id.cf1a=ebf073c7-a54f-470d-9bbf-f26fa5da69d6.1719224217.22.1720000401.1719841079.66812e7c-694c-4c6f-b880-85f58a5ef5fc; _ga_YVVRYGL0E0=GS1.1.1719996956.28.1.1720000401.48.0.0",
        "origin": "https://www.tradingview.com",
        "priority": "u=1, i",
        "referer": "https://www.tradingview.com/",
        "sec-ch-ua": "\"Not/A)Brand\";v=\"8\", \"Chromium\";v=\"126\", \"Google Chrome\";v=\"126\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    }

    # Data payload
    data = {
        "filter": [
            {"left": "expiration", "operation": "nempty"},
            {"left": "subtype", "operation": "nequal", "right": "continuous"}
        ],
        "index_filters": [
            {"name": "root", "values": [ticker]}
        ],
        "markets": ["futures"],
        "columns": ["expiration", "close", "pricescale", "minmov"],
        "sort": {"sortBy": "expiration", "sortOrder": "asc"}
    }

    # Convert the data payload to JSON formatted string
    data_json = json.dumps(data)

    # Make the POST request
    response = requests.post(url, headers=headers, data=data_json)
   
    response_data = response.json()

    # Create a DataFrame from the response data
    data_list = []
    for item in response_data['data']:
        row = {
            'Symbol': item['s'],
            'Expiration': item['d'][0],
            'Close': item['d'][1],
            'Price Scale': item['d'][2],
            'Min Move': item['d'][3]
        }
        data_list.append(row)

    df = pd.DataFrame(data_list)
    df['Expiration'] = pd.to_datetime(df['Expiration'], format='%Y%m%d')
    return df


def plot_forward_curve(options):
    options = options.sort_values(by='Expiration')
    expiration_dates = options['Expiration']
    prices = options['Close']

    plt.figure(figsize=(10, 6))
    plt.plot(expiration_dates, prices, marker='o', linestyle='-', color='b')
    plt.title('Forward Curve')
    plt.xlabel('Expiration Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

plot_forward_curve(forward_request("NYMEX:CL"))