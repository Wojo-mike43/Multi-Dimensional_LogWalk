# Multi-Dimensional_LogWalk
This project is a multi-dimensional lognormal random walk that models the evolution of a multi-asset portfolio over time. In this simulation, each asset's price path is influenced by historical returns, volatility, and covariances among the assets in the portfolio. The project utilizes `yfinance`, `pandas`, `numpy`, to perform the calculations, while `matplotlib` and `streamlit` are used for visualizations and a user interface.

## How It Works:
- **Overview:**
  A multi-dimensional random walk is a stochastic model that simulates the paths of multiple correlated assets, with each following a log-normal distribution. The model considers each asset's average growth rate (drift) and volatility. The interconnected movements of assets through a covariance matrix reflect how each asset’s price movements influence others in the portfolio. This allows the stocks in the portfolio to have both shared and independent volatilities, mimicking a real-world portfolio’s dynamic nature.


- **Data:**  
  A portfolio of stocks is pre-loaded into the script using a Python list. This portfolio consists of 24 stocks that used to be held in the student-run growth portfolio I managed from 2022 - 2023. `yfinance` is utilized to collect historical data on the stocks over the past three years.

- **Lognormal Random Walk:**  
Using `numpy` and `pandas`, the lognormal returns for each stock are calculated, along with each stock’s historical volatility and drift coefficient (average daily growth). Next, a covariance matrix is constructed for the portfolio. This covariance matrix, combined with each stock's volatility and drift coefficient, is used to generate a series of random lognormal returns using `numpy’s` `np.random.multivariate_normal`. This function produces random returns based on the statistical properties of the assets, simulating their paths over time.

- **Portfolio Statistics:**  
  Basic portfolio statistics such as portfolio growth, annual mean volatility, and annual mean return are calculated.

- **GUI:**  
  The project is wrapped in a `streamlit` GUI for easy usability and demonstration purposes. Here, the sample statistics mentioned above are displayed as well as the covariance matrix for the portfolio. `matplotlib` is used to plot sample price paths for each stock in the random walk.

- **Streamlit Project Link:**
  [https://multi-dimensionallogwalk-nhv765us2texyjhumeqgfa.streamlit.app/]
  
  
