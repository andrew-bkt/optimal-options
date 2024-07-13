# data/data_fetcher.py
import yfinance as yf
import pandas as pd
from datetime import datetime

class OptionDataFetcher:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.stock = yf.Ticker(self.ticker)

    def fetch_data(self):
        expirations = self.stock.options
        today = datetime.now()
        valid_expirations = [exp for exp in expirations if (datetime.strptime(exp, '%Y-%m-%d') - today).days >= 30]
        if not valid_expirations:
            raise ValueError("No valid expiration dates found")
        nearest_expiration = min(valid_expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - today))
        print(f"Using expiration date: {nearest_expiration}")
        options = self.stock.option_chain(nearest_expiration)
        calls = options.calls
        
        try:
            underlying = self.stock.history(start=self.start_date, end=self.end_date, interval="1h")
            if len(underlying) == 0:
                raise ValueError("No hourly data available")
        except:
            print(f"Hourly data not available for {self.ticker}. Falling back to daily data.")
            underlying = self.stock.history(start=self.start_date, end=self.end_date)
        
        if len(underlying) == 0:
            raise ValueError(f"No data available for {self.ticker}")

        underlying_info = self.stock.info
        underlying['volume'] = underlying_info.get('volume', 0)
        underlying['market_cap'] = underlying_info.get('marketCap', 0)
        underlying['sector'] = underlying_info.get('sector', 'Unknown')
        
        print(f"Number of call options: {len(calls)}")
        print(f"Number of periods in underlying data: {len(underlying)}")
        return calls, underlying, nearest_expiration

    def fetch_market_data(self):
        sp500 = yf.Ticker('^GSPC')
        vix = yf.Ticker('^VIX')
        try:
            market_data = pd.DataFrame({
                'sp500': sp500.history(start=self.start_date, end=self.end_date, interval="1h")['Close'],
                'vix': vix.history(start=self.start_date, end=self.end_date, interval="1h")['Close']
            })
        except:
            print("Hourly market data not available. Falling back to daily data.")
            market_data = pd.DataFrame({
                'sp500': sp500.history(start=self.start_date, end=self.end_date)['Close'],
                'vix': vix.history(start=self.start_date, end=self.end_date)['Close']
            })
        return market_data


    def fetch_fundamental_data(self):
        fundamentals = self.stock.info
        return {
            'pe_ratio': fundamentals.get('trailingPE', None),
            'dividend_yield': fundamentals.get('dividendYield', None),
            'beta': fundamentals.get('beta', None),
        }

