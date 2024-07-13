# options_screening/features/advanced_features.py
from .base_feature_engineer import BaseFeatureEngineer
import numpy as np
from scipy.stats import norm

class AdvancedFeatureEngineer(BaseFeatureEngineer):
    def __init__(self, calls, underlying, expiration_date, risk_free_rate=0.05):
        super().__init__(calls, underlying, expiration_date)
        self.risk_free_rate = risk_free_rate

    def engineer_features(self):
        self._calculate_implied_volatility()
        self._calculate_greeks()
        self._calculate_price_ratios()
        self._calculate_volatility_ratios()
        
        return self.calls

    def calculate_implied_volatility(self):
        # This is a simplified IV calculation. For real-world use, consider using a more robust method.
        self.calls['implied_volatility'] = np.sqrt(2 * np.pi / self.calls['time_to_expiry']) * (self.calls['lastPrice'] / self.calls['strike'])

    def calculate_greeks(self):
        S = self.underlying['Close'].iloc[-1]
        K = self.calls['strike']
        T = self.calls['time_to_expiry'] / 365
        r = self.risk_free_rate
        sigma = self.calls['implied_volatility']

        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)

        self.calls['delta'] = norm.cdf(d1)
        self.calls['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        self.calls['theta'] = -(S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))) - r * K * np.exp(-r*T) * norm.cdf(d2)
        self.calls['vega'] = S * norm.pdf(d1) * np.sqrt(T)

    def calculate_price_ratios(self):
        self.calls['price_to_strike'] = self.calls['lastPrice'] / self.calls['strike']
        self.calls['price_to_underlying'] = self.calls['lastPrice'] / self.underlying['Close'].iloc[-1]

    def calculate_moneyness(self):
        self.calls['moneyness'] = np.log(self.underlying['Close'].iloc[-1] / self.calls['strike'])

    def calculate_time_to_earnings(self, next_earnings_date):
        self.calls['time_to_earnings'] = (pd.to_datetime(next_earnings_date) - pd.to_datetime(self.calls['lastTradeDate'])).dt.days

    def calculate_volatility_ratios(self):
        historical_vol = self.underlying['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
        self.calls['iv_to_hv_ratio'] = self.calls['implied_volatility'] / historical_vol.iloc[-1]

    def calculate_open_interest_ratios(self):
        self.calls['oi_to_volume_ratio'] = self.calls['openInterest'] / self.calls['volume'].replace(0, 1)

    def engineer_features(self, next_earnings_date=None):
        self.calculate_implied_volatility()
        self.calculate_greeks()
        self.calculate_price_ratios()
        self.calculate_moneyness()
        self.calculate_volatility_ratios()
        self.calculate_open_interest_ratios()
        
        if next_earnings_date:
            self.calculate_time_to_earnings(next_earnings_date)
        
        return self.calls

def get_feature_names():
    return [
        'implied_volatility', 'delta', 'gamma', 'theta', 'vega',
        'price_to_strike', 'price_to_underlying', 'moneyness',
        'iv_to_hv_ratio', 'oi_to_volume_ratio'
    ]