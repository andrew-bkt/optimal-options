from .basic_features import BasicFeatureEngineer
from .technical_features import TechnicalFeatureEngineer
from .advanced_features import AdvancedFeatureEngineer
from .profit_target import ProfitTargetEngineer
from .delta_profit_target import DeltaProfitTargetEngineer

class FeatureFactory:
    @staticmethod
    def create_feature_engineer(feature_type, calls, underlying, expiration_date, **kwargs):
        if feature_type == 'basic':
            return BasicFeatureEngineer(calls, underlying, expiration_date)
        elif feature_type == 'technical':
            return TechnicalFeatureEngineer(calls, underlying, expiration_date)
        elif feature_type == 'advanced':
            return AdvancedFeatureEngineer(calls, underlying, expiration_date, **kwargs)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    @staticmethod
    def create_target_engineer(target_type, calls, underlying, **kwargs):
        if target_type == 'profit':
            return ProfitTargetEngineer(calls, underlying, profit_threshold=kwargs.get('profit_threshold', 0.005))
        elif target_type == 'delta_profit':
            return DeltaProfitTargetEngineer(calls, underlying, 
                                             profit_threshold=kwargs.get('profit_threshold', 0.005),
                                             delta_threshold=kwargs.get('delta_threshold', 0.5))
        else:
            raise ValueError(f"Unknown target type: {target_type}")
