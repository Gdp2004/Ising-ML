from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os

class ModelTrainer:
    def __init__(self, model_type='logistic'):
        self.model_type = model_type
        self.model = self._get_model()

    def _get_model(self):
        if self.model_type == 'logistic':
            # Simple linear model for Physics features
            return LogisticRegression(max_iter=1000)
        elif self.model_type == 'svm':
            # SVM for Physics features
            return SVC(probability=True)
        elif self.model_type == 'rf':
            # Random Forest for Raw Data
            return RandomForestClassifier(n_estimators=100)
        elif self.model_type == 'mlp':
            # MLP for Raw Data
            return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        print(f"Training {self.model_type}...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def predict_proba(self, X_test):
        """Return probability estimates for the test data."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_test)
        else:
            raise NotImplementedError(f"Model {self.model_type} does not support predict_proba")

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return acc, cm, report
    
    def save_model(self, filename):
        """Save the trained model to disk"""
        print(f"Saving model to {filename}...")
        joblib.dump(self.model, filename)
        print("Model saved.")
    
    @staticmethod
    def load_model(filename, model_type='logistic'):
        """Load a trained model from disk"""
        if not os.path.exists(filename):
            print(f"Model file {filename} not found.")
            return None
        print(f"Loading model from {filename}...")
        trainer = ModelTrainer(model_type)
        trainer.model = joblib.load(filename)
        return trainer
