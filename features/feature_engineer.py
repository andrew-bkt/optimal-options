# options_screening/features/feature_engineer.py

import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger import app_logger as logger

class FeatureEngineer:
    def __init__(self, calls, underlying, expiration_date):
        self.calls = calls
        self.underlying = underlying
        self.expiration_date = expiration_date

    def engineer_features(self):
        logger.info("Starting feature engineering...")
        logger.debug(f"Columns in calls data: {self.calls.columns}")
        
        current_price = self.underlying['Close'].iloc[-1]
        self.calls['moneyness'] = current_price / self.calls['strike']
        self.calls['time_to_expiry'] = (datetime.strptime(self.expiration_date, '%Y-%m-%d') - datetime.now()).days
        
        self.underlying['returns'] = self.underlying['Close'].pct_change()
        self.calls['historical_volatility'] = self.underlying['returns'].rolling(window=30).std() * np.sqrt(252)
        
        self.calls['volume_oi_ratio'] = self.calls['volume'] / self.calls['openInterest'].replace(0, 1)
        
        logger.debug(f"Columns after feature engineering: {self.calls.columns}")
        logger.info(f"Number of rows after feature engineering: {len(self.calls)}")
        return self.calls

    def create_target(self, calls_with_features):
        logger.info("Creating target variable...")
        logger.debug(f"Columns in calls_with_features: {calls_with_features.columns}")
        
        days_to_expiration = (datetime.strptime(self.expiration_date, '%Y-%m-%d') - datetime.now()).days
        forward_days = min(days_to_expiration // 4, 10)  # Increased from 5 to 10
        logger.info(f"Using {forward_days} days for forward-looking period")
        
        future_prices = self.underlying['Close'].iloc[-1-forward_days:]
        
        if len(future_prices) < forward_days:
            logger.warning(f"Not enough future price data. Available: {len(future_prices)}, Required: {forward_days}")
            return pd.DataFrame()
        
        future_price = future_prices.iloc[-1]
        current_price = self.underlying['Close'].iloc[-1]
        
        calls_with_features['potential_profit'] = (future_price - calls_with_features['strike']).clip(lower=0) - calls_with_features['lastPrice']
        calls_with_features['profit_percentage'] = calls_with_features['potential_profit'] / calls_with_features['lastPrice']
        
        calls_with_features['target'] = calls_with_features['profit_percentage'] > 0.005  # Reduced from 0.01 to 0.005 (0.5% profit)
        
        logger.debug(f"Columns after creating target: {calls_with_features.columns}")
        logger.info(f"Number of rows after creating target: {len(calls_with_features)}")
        logger.info(f"Number of positive targets: {calls_with_features['target'].sum()}")
        logger.info(f"Target distribution: {calls_with_features['target'].value_counts(normalize=True)}")
        
        return calls_with_features