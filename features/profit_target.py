from .base_feature_engineer import BaseTargetEngineer

class ProfitTargetEngineer(BaseTargetEngineer):
    def __init__(self, calls, underlying, profit_threshold=0.005):
        super().__init__(calls, underlying)
        self.profit_threshold = profit_threshold

    def create_target(self):
        current_price = self.underlying['Close'].iloc[-1]
        self.calls['potential_profit'] = (current_price - self.calls['strike']).clip(lower=0) - self.calls['lastPrice']
        self.calls['profit_percentage'] = self.calls['potential_profit'] / self.calls['lastPrice']
        self.calls['target'] = self.calls['profit_percentage'] > self.profit_threshold
        return self.calls
