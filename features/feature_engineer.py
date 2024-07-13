# features/feature_engineer.py
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
        logger.debug(f"Underlying data shape: {self.underlying.shape}")
        
        try:
            current_price = self.underlying['Close'].iloc[-1]
            self.calls['moneyness'] = current_price / self.calls['strike']
            self.calls['time_to_expiry'] = (datetime.strptime(self.expiration_date, '%Y-%m-%d') - datetime.now()).days
            
            self.underlying['returns'] = self.underlying['Close'].pct_change()
            
            for window in [10, 30, 60]:
                self.calls[f'historical_volatility_{window}d'] = self.underlying['returns'].rolling(window=window).std() * np.sqrt(252)
            
            self.calls['volume_oi_ratio'] = self.calls['volume'] / self.calls['openInterest'].replace(0, 1)
            
            logger.debug(f"Columns after feature engineering: {self.calls.columns}")
            logger.info(f"Number of rows after feature engineering: {len(self.calls)}")
        except Exception as e:
            logger.error(f"Error in engineer_features: {str(e)}")
            raise
        
        return self.calls

    def create_target(self, calls_with_features):
        logger.info("Creating target variable...")
        logger.debug(f"Columns in calls_with_features: {calls_with_features.columns}")
        
        try:
            current_price = self.underlying['Close'].iloc[-1]
            previous_price = self.underlying['Close'].iloc[-2]
            
            calls_with_features['potential_profit'] = (current_price - calls_with_features['strike']).clip(lower=0) - calls_with_features['lastPrice']
            calls_with_features['profit_percentage'] = calls_with_features['potential_profit'] / calls_with_features['lastPrice']
            
            calls_with_features['target'] = calls_with_features['profit_percentage'] > 0.005  # 0.5% profit threshold
            
            logger.debug(f"Columns after creating target: {calls_with_features.columns}")
            logger.info(f"Number of rows after creating target: {len(calls_with_features)}")
            logger.info(f"Number of positive targets: {calls_with_features['target'].sum()}")
            logger.info(f"Target distribution: {calls_with_features['target'].value_counts(normalize=True)}")
        except Exception as e:
            logger.error(f"Error in create_target: {str(e)}")
            raise
        
        return calls_with_features
