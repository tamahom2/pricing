import numpy as np

def monte_carlo_call_down_and_in(S, X, H, T, r, v, num_simulations=10000, num_steps=100):
    dt = T / num_steps
    discount_factor = np.exp(-r * T)
    
    # Simulate paths using the risk-free rate in the drift term
    S_paths = np.zeros((num_simulations, num_steps + 1))
    S_paths[:, 0] = S
    for t in range(1, num_steps + 1):
        Z = np.random.standard_normal(num_simulations)
        # Adjust the drift term with the risk-free rate r
        S_paths[:, t] = S_paths[:, t-1] * np.exp((r - 0.5 * v**2) * dt + v * np.sqrt(dt) * Z)
    
    # Check barrier condition for down-and-in options
    barrier_crossed = np.min(S_paths, axis=1) <= H
    
    # Calculate payoffs
    payoffs = np.maximum(S_paths[:, -1] - X, 0)
    payoffs[~barrier_crossed] = 0  # Set payoff to zero if barrier not crossed
    
    # Discount payoffs to present value
    discounted_payoffs = discount_factor * payoffs
    
    # Calculate the option price as the mean of discounted payoffs
    option_price = np.mean(discounted_payoffs)
    return option_price

# Run the simulation for H = 115 and H = 130
np.random.seed(0)  # Set seed for reproducibility
option_price_cdi_H115 = monte_carlo_call_down_and_in(100, 100, 101, 0.5, 0.08, 0.2, num_simulations=100000)
option_price_cdi_H130 = monte_carlo_call_down_and_in(100, 100, 130, 0.5, 0.08, 0.2, num_simulations=100000)

import pandas as pd

# Define the new data
data = {
    "Call Prices": [0.044751875020475694, 0.0667013821726324, 0.08371107948231507, 0.0971210102569191, 0.10820857114341774, 0.11920755632107259, 0.12908363700945635, 0.136102784397675, 0.14546576655772583, 0.15299581979874413, 0.1598140269138797, 0.1686126158651884, 0.1752510332124427, 0.18210903021654676, 0.18797419630047305, 0.1934002951327356, 0.19963806129692438, 0.20568211046781018, 0.21134380729393182, 0.21642534491051074, 0.2215662473526765, 0.22472392992116957, 0.23149100858054666, 0.23612828463515081],
    "Forward Prices": [0.9792629606495941, 0.9657713928794504, 0.9552779512804498, 0.9464084946908183, 0.9397876327295441, 0.9335415365396628, 0.9270455965021861, 0.9240474703310431, 0.9179262960649595, 0.9129294191130544, 0.9095565271705185, 0.9025608994378513, 0.8981886321049345, 0.8940662086196127, 0.8904434728294817, 0.8875702685821362, 0.8828232354778264, 0.8788257339163023, 0.8750780762023735, 0.8715802623360399, 0.868582136164897, 0.868207370393504, 0.8623360399750156, 0.859212991880075],
    "Volatility": [0.3] * 24,
    "Expiration": ["20240820", "20240920", "20241022", "20241120", "20241219", "20250121", "20250220", "20250320", "20250422", "20250520", "20250620", "20250722", "20250820", "20250922", "20251021", "20251120", "20251219", "20260120", "20260220", "20260320", "20260421", "20260519", "20260622", "20260721"]
}

# Create a DataFrame
df = pd.DataFrame(data)
df['Expiration'] = pd.to_datetime(df['Expiration'], format='%Y%m%d')
# Save the DataFrame to an Excel file
output_file_path = 'MatrixData.xlsx'
df.to_excel(output_file_path, index=False)