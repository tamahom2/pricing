from datetime import date
from dateutil.relativedelta import relativedelta

# Hard-coded rates for simplicity
RATES = {
    '1_month': 5.44,
    '2_month': 5.46,
    '3_month': 5.50,
    '4_month': 5.46,
    '6_month': 5.36,
    '1_year': 5.13,
    '2_year': 4.71,
    '3_year': 4.53,
    '5_year': 4.32,
    '7_year': 4.32,
    '10_year': 4.32,
    '20_year': 4.55,
    '30_year': 4.45
}


def get_risk_free_rate(expiration_date):
    # Define today's date
    today_date = date.today()
    
    # Calculate the difference between today's date and the expiration date
    delta = relativedelta(expiration_date, today_date)
    rate = 0 
    if delta.years >= 20:
        rate = RATES['20_year']
    elif delta.years >= 10:
        rate = RATES['10_year']
    elif delta.years >= 7:
        rate = RATES['7_year']
    elif delta.years >= 5:
        rate = RATES['5_year']
    elif delta.years >= 3:
        rate = RATES['3_year']
    elif delta.years >= 2:
        rate = RATES['2_year']
    elif delta.years >= 1:
        rate = RATES['1_year']
    elif delta.months >= 6:
        rate = RATES['6_month']
    elif delta.months >= 4:
        rate = RATES['4_month']
    elif delta.months >= 3:
        rate = RATES['3_month']
    elif delta.months >= 2:
        rate = RATES['2_month']
    else:
        rate = RATES['1_month']
    return rate/100