
import pickle
import joblib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import os
import logging

logger = logging.getLogger(__name__)

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_count = 20
    
    def load_model(self):
        model_path = "model.pkl"
        scaler_path = "scaler.pkl"
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Model loaded from disk")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self._train_dummy_model()
        else:
            logger.info("No saved model found, training dummy model")
            self._train_dummy_model()
    
    def _train_dummy_model(self):
        logger.info("Training dummy model...")
        X, y = make_classification(
            n_samples=1000, 
            n_features=self.feature_count,
            n_informative=15,
            n_redundant=5,
            random_state=42
        )
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.model.fit(X_scaled, y)
        joblib.dump(self.model, "model.pkl")
        joblib.dump(self.scaler, "scaler.pkl")
        accuracy = self.model.score(X_scaled, y)
        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
    
    def predict(self, features):
        if self.model is None:
            raise ValueError("Model not loaded")
        if self.scaler:
            features = self.scaler.transform(features)
        return self.model.predict(features)
    
    def is_loaded(self):
        return self.model is not None