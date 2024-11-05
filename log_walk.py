import pandas as pd
import yfinance as yf
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

preloaded_portfolio = [
    "ADBE", "AMZN", "AXP", "CELH", "CHX", "LNG", "EL", "XOM", "GS", "HON",
    "JHX", "JAZZ", "LHX", "MRO", "MSFT", "MNST", "NEE", "PFE", "PWR",
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
    mu = log_returns.mean() + .5 * log_returns.var()
    sigma = log_returns.std()
    cov_matrix = log_returns.cov()

    steps = 365
    time_horizon = 1

    random_numbers = np.random.multivariate_normal(mean=mu, cov=cov_matrix, size=steps)

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
    st.title('Multi-Dimensional Lognormal Random Walk')

    with st.sidebar:
        st.title('Random Walk Parameters: ')
        button = st.button('Run Analysis')

        st.markdown(
            """
            <div style="max-height: 500px; overflow-y: auto; padding: 5px; border: 1px solid #ccc; border-radius: 5px;">
                <strong>Default Portfolio Stocks:</strong>
                <ul style="list-style-type: none; padding-left: 10px;">
            """ +
            "".join([f"<li>• <strong>{stock}</strong></li>" for stock in preloaded_portfolio]) +
            """
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Personal Information: ")
        linkedin_url = 'https://www.linkedin.com/in/michaelwojciechowski93'
        st.markdown(
            f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">Michael Wojciechowski</a>',
            unsafe_allow_html=True)


    if button == True:

        stock_data = data_pull(days=1095, stocks=preloaded_portfolio)
        sim_data = log_walk(data=stock_data)
        port_mean_return, port_annual_vol, port_correl_matrix = port_analysis(port_data=sim_data)

        st.markdown('### Statistics: ')

        port_gain = round(sim_data.iloc[-1].mean() - sim_data.iloc[0].mean(), 2)
        port_pct_gain = round(port_gain / sim_data.iloc[0].mean(), 2)


        col1, col2, = st.columns(2)
        col1.metric(label='Portfolio Gain: ', value= f"${port_gain}")
        col2.metric(label='Portfolio Percentage Gain: ', value=f"{port_pct_gain}%")
        col1.metric(label='Portfolio Annual Mean Return: ', value=f"{round(port_mean_return * 100, 2)}%")
        col2.metric(label='Portfolio Annual Volatility: ', value=f"{round(port_annual_vol * 100, 2)}%")

        st.markdown('### Asset Correlation Maxtrix: ')
        st.dataframe(round(port_correl_matrix,2))


        #Plot
        st.markdown('### Lognormal Random Walk Plot: ')

        plt.style.use('dark_background')
        plt.figure(figsize=(10, 8))
        plt.plot(sim_data, linestyle='-.', linewidth=1.5, label=sim_data.columns)

        plt.title('Random Walk Results')
        plt.xlabel('Time')
        plt.ylabel('Stock Prices')

        plt.legend(sim_data.columns, loc='center left', fontsize='small')
        plt.tight_layout()
        plt.grid(color='grey')

        st.pyplot(plt)

        # Model Description
        st.markdown('#### About the Project')
        with st.expander("Click Here To Read More About The Project:"):
            st.write("Test")
