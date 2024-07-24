import requests
from bs4 import BeautifulSoup

quotes = ["agricultural","energy","metals"]

# URL to scrape
for quote in quotes:
    url = f"https://www.tradingview.com/markets/futures/quotes-{quote}/"

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the HTML content of the page
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all <tr> elements with the 'data-rowkey' attribute
        rows = soup.find_all('tr', attrs={"data-rowkey": True})
        
        # Extract the 'data-rowkey' values
        data_rowkeys = [row['data-rowkey'] for row in rows]
        
        # Print the extracted data-rowkey values
        for rowkey in data_rowkeys:
            print(rowkey)
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")