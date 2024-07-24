from datetime import datetime

MONTHS = ["F","G","H","J","K","M","N","Q","U","V","X","Z"]

def future_ticker(ticker, years=4):
    res = []
    current_date = datetime.now()
    current_year = current_date.year
    current_month = current_date.month
    
    # Iterate through the next 'years' years
    for i in range(years):
        for j in range(12):
            # Calculate the year and month index
            year = current_year + i + (current_month + j - 1) // 12
            month_index = (current_month + j - 1) % 12
            
            # Append the ticker with the corresponding month and year
            res.append(f"{ticker}{MONTHS[month_index]}{year}")
            
            # Stop if we've reached the last month in the range
            if year == current_year + years and month_index == 11:
                break
    
    return res