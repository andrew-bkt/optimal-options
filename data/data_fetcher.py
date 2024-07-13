# data/data_fetcher.py
import yfinance as yf
from datetime import datetime

class OptionDataFetcher:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        stock = yf.Ticker(self.ticker)
        expirations = stock.options
        today = datetime.now()
        valid_expirations = [exp for exp in expirations if (datetime.strptime(exp, '%Y-%m-%d') - today).days >= 30]
        if not valid_expirations:
            raise ValueError("No valid expiration dates found")
        nearest_expiration = min(valid_expirations, key=lambda x: abs(datetime.strptime(x, '%Y-%m-%d') - today))
        print(f"Using expiration date: {nearest_expiration}")
        options = stock.option_chain(nearest_expiration)
        calls = options.calls
        underlying = stock.history(start=self.start_date, end=self.end_date)
        print(f"Number of call options: {len(calls)}")
        print(f"Number of days in underlying data: {len(underlying)}")
        return calls, underlying, nearest_expiration