from abc import ABC, abstractmethod

class BaseFeatureEngineer(ABC):
    def __init__(self, calls, underlying, expiration_date):
        self.calls = calls
        self.underlying = underlying
        self.expiration_date = expiration_date

    @abstractmethod
    def engineer_features(self):
        pass

class BaseTargetEngineer(ABC):
    def __init__(self, calls, underlying):
        self.calls = calls
        self.underlying = underlying

    @abstractmethod
    def create_target(self):
        pass
