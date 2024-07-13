# features/delta_profit_target.py

from .base_feature_engineer import BaseTargetEngineer

class DeltaProfitTargetEngineer(BaseTargetEngineer):
    def __init__(self, calls, underlying, profit_threshold=0.005, delta_threshold=0.5):
        super().__init__(calls, underlying)
        self.profit_threshold = profit_threshold
        self.delta_threshold = delta_threshold

    def create_target(self):
        current_price = self.underlying['Close'].iloc[-1]
        self.calls['potential_profit'] = (current_price - self.calls['strike']).clip(lower=0) - self.calls['lastPrice']
        self.calls['profit_percentage'] = self.calls['potential_profit'] / self.calls['lastPrice']
        
        # Assuming 'delta' is already calculated in the calls DataFrame
        self.calls['target'] = (self.calls['profit_percentage'] > self.profit_threshold) & (self.calls['delta'] > self.delta_threshold)
        
        return self.calls