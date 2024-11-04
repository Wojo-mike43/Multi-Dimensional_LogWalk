import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

stocks = [
    "ADBE", "AMZN", "AXP", "CELH", "CHX", "LNG", "EL", "XOM", "GS", "HON",
    "JHX", "JAZZ", "LHX", "LLY", "MRO", "MSFT", "MNST", "NEE", "PFE", "PWR",
    "SNEX", "TMUS", "BLD", "MODG", "V"
]

def data_pull(days, stocks):
    end = dt.datetime.today() - dt.timedelta(days=1)
    start = end - dt.timedelta(days)

    tickers = yf.Tickers(stocks)
    data = tickers.download(start=start, end=end, period='1d')['Close']

    data = data.reset_index()
    data.columns.name = None

    data = data.set_index('Date')

    return data


def log_walk(data):
    log_returns = np.log(data / data.shift(1)).dropna()
    mu = log_returns.mean() + .5 * log_returns.var()  # to account for extra growth due to volatility
    sigma = log_returns.std()
    cov_matrix = log_returns.cov()

    # Generate Random Numbers
    random_numbers = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=steps)  # 365 rows by 25 col

    # lognormal_walk
    mu = mu / 252
    sigma = sigma / np.sqrt(252)

    time_step = steps / time_horizon
    last_prices = data.iloc[-1]

    log_walk = pd.DataFrame(np.zeros((steps + 1, len(data.columns))), columns=data.columns)
    log_walk.iloc[0] = last_prices

    for t in range(1, steps + 1):
        log_walk.iloc[t] = log_walk.iloc[t - 1] * np.exp(
            (mu - .5 * sigma ** 2) * time_step + sigma * np.sqrt(time_step) * random_numbers[t - 1])

    return log_walk


def port_analysis(port_data):
    simulation_returns = port_data.pct_change().dropna()

    mean_return = simulation_returns.mean().mean() * 252

    simulation_vol = simulation_returns.std()
    annual_vol = (simulation_vol * np.sqrt(252)).mean()

    correl_matrix = simulation_returns.corr()

    return mean_return, annual_vol, correl_matrix




if __name__ == '__main__':
    # parameters#
    days = 365 * 3
    steps = 365
    time_horizon = 1


    stock_data = data_pull(days=days, stocks=stocks)
    sim_data = log_walk(data=stock_data)
    port_mean_return, port_annual_vol, port_correl_matrix = port_analysis(port_data=sim_data)


    print(f"Portfolio Annual Mean Return: {round(port_mean_return * 100, 2)}%")
    print(f"Portfolio Annual Volatility: {round(port_annual_vol * 100, 2)}%")
    print(f"Asset Correlation Matrix:: {round(port_correl_matrix, 2)}")






    #Plot
    #plt.figure(figsize=(10, 8))
    #plt.plot(log_walk)

    #plt.title('Random Walk Results')
    #plt.xlabel('Time')
    #plt.ylabel('Stock Prices')
    #plt.grid()

    #plt.legend(log_walk.columns, loc='center left', fontsize='small')
    #plt.tight_layout()

    #plt.show()
