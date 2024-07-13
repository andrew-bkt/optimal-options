# data/data_pipeline.py
import pandas as pd
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .data_fetcher import OptionDataFetcher
from features.feature_factory import FeatureFactory
from utils.logger import app_logger as logger

class DataPipeline:
    def __init__(self, config):
        self.config = config

    def process_data(self):
        all_data = []
        for ticker in self.config.get_nested('data', 'tickers'):
            data = self._process_single_ticker(ticker)
            if data is not None:
                all_data.append(data)
        if not all_data:
            raise ValueError("No valid data available for any of the provided tickers.")
        return pd.concat(all_data, ignore_index=True)

    def _process_single_ticker(self, ticker):
        try:
            logger.info(f"Processing data for ticker: {ticker}")
            start_date = self.config.get_nested('data', 'start_date')
            end_date = self.config.get_nested('data', 'end_date')
            
            if end_date == 'auto':
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            data_fetcher = OptionDataFetcher(ticker, start_date, end_date)
            calls, underlying, expiration_date = data_fetcher.fetch_data()
            
            logger.debug(f"Shape of calls data: {calls.shape}")
            logger.debug(f"Shape of underlying data: {underlying.shape}")
            logger.debug(f"Columns in calls data: {calls.columns}")
            logger.debug(f"Columns in underlying data: {underlying.columns}")
            
            if calls.empty or underlying.empty:
                logger.warning(f"No valid data for {ticker}. Skipping this ticker.")
                return None
            
            feature_types = self.config.get_nested('features', 'types')
            for feature_type in feature_types:
                feature_engineer = FeatureFactory.create_feature_engineer(feature_type, calls, underlying, expiration_date)
                calls = feature_engineer.engineer_features()

            target_config = self.config.get('target')
            if target_config['type'] == 'delta_profit' and 'delta' not in calls.columns:
                advanced_engineer = FeatureFactory.create_feature_engineer('advanced', calls, underlying, expiration_date)
                calls = advanced_engineer.engineer_features()

            target_engineer = FeatureFactory.create_target_engineer(target_config['type'], calls, underlying, **target_config.get('params', {}))
            calls_with_target = target_engineer.create_target()
            
            if len(calls_with_target) > 0:
                calls_with_target['ticker'] = ticker
                logger.info(f"Successfully processed data for {ticker}")
                return calls_with_target
            else:
                logger.warning(f"No valid data for {ticker} after processing. Skipping this ticker.")
                return None
        
        except Exception as e:
            logger.error(f"An error occurred while processing {ticker}: {str(e)}")
            return None


            
    def preprocess_data(self, data):
        feature_columns = (
            self.config.get_nested('features', 'basic') +
            self.config.get_nested('features', 'technical') +
            self.config.get_nested('features', 'advanced') +
            self.config.get_nested('features', 'fundamental')
        )
        
        # Check which columns are actually present in the data
        available_columns = [col for col in feature_columns if col in data.columns]
        missing_columns = set(feature_columns) - set(available_columns)
        
        if missing_columns:
            logger.warning(f"The following columns are missing from the data: {missing_columns}")
        
        X = data[available_columns]
        y = data['target']
        
        # Check for NaN values
        nan_columns = X.columns[X.isna().any()].tolist()
        if nan_columns:
            logger.warning(f"Columns with NaN values: {nan_columns}")
            logger.warning(f"Number of NaN values:\n{X.isna().sum()}")
        
        # Check for columns with all NaN values
        all_nan_columns = X.columns[X.isna().all()].tolist()
        if all_nan_columns:
            logger.warning(f"Columns with all NaN values: {all_nan_columns}")
            logger.warning("These columns will be dropped.")
            X = X.drop(columns=all_nan_columns)
        
        # Impute NaN values
        imputer = SimpleImputer(strategy=self.config.get_nested('preprocessing', 'imputer_strategy'))
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X.columns)
        
        return X_scaled, y
