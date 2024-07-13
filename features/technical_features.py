from .base_feature_engineer import BaseFeatureEngineer
import ta

class TechnicalFeatureEngineer(BaseFeatureEngineer):
    def engineer_features(self):
        self.calls['rsi'] = ta.momentum.RSIIndicator(self.underlying['Close']).rsi().iloc[-1]
        
        macd = ta.trend.MACD(self.underlying['Close'])
        self.calls['macd'] = macd.macd().iloc[-1]
        self.calls['macd_signal'] = macd.macd_signal().iloc[-1]
        
        bollinger = ta.volatility.BollingerBands(self.underlying['Close'])
        self.calls['bollinger_high'] = bollinger.bollinger_hband().iloc[-1]
        self.calls['bollinger_low'] = bollinger.bollinger_lband().iloc[-1]
        
        self.calls['moving_average_50'] = ta.trend.SMAIndicator(self.underlying['Close'], window=50).sma_indicator().iloc[-1]
        self.calls['moving_average_200'] = ta.trend.SMAIndicator(self.underlying['Close'], window=200).sma_indicator().iloc[-1]
        
        return self.calls
