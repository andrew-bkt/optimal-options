from .base_feature_engineer import BaseFeatureEngineer
from datetime import datetime
import numpy as np

class BasicFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self):
        current_price = self.underlying['Close'].iloc[-1]
        self.calls['moneyness'] = current_price / self.calls['strike']
        self.calls['time_to_expiry'] = (datetime.strptime(self.expiration_date, '%Y-%m-%d') - datetime.now()).days
        
        self.underlying['returns'] = self.underlying['Close'].pct_change()
        
        for window in [10, 30, 60]:
            self.calls[f'historical_volatility_{window}d'] = self.underlying['returns'].rolling(window=window).std() * np.sqrt(252)
        
        self.calls['volume_oi_ratio'] = self.calls['volume'] / self.calls['openInterest'].replace(0, 1)
        
        return self.calls
