"""
Classifier module for radar sea clutter classification.
Includes traditional ML (Random Forest, SVM) and deep learning (LSTM, CNN) models.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Union
import pickle
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class RadarClassifier:
    """
    Unified classifier for radar clutter detection.
    Supports traditional ML and deep learning approaches.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model ('random_forest', 'svm', 'lstm', 'cnn')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the specified model."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
        elif self.model_type in ['lstm', 'cnn']:
            # Deep learning models will be built dynamically
            self.model = None
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_data(self, features: np.ndarray, labels: np.ndarray, 
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        
        Args:
            features: Feature matrix
            labels: Target labels
            test_size: Fraction of data for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, y_encoded, test_size=test_size, 
            random_state=42, stratify=y_encoded
        )
        
        # Scale features for traditional ML models
        if self.model_type in ['random_forest', 'svm']:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    def build_lstm_model(self, input_shape: Tuple[int, int], num_classes: int = 2) -> keras.Model:
        """
        Build LSTM model for time series classification.
        
        Args:
            input_shape: Input shape (timesteps, features)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.LSTM(128, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.3),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def build_cnn_model(self, input_shape: Tuple[int, int, int], num_classes: int = 2) -> keras.Model:
        """
        Build CNN model for 2D spectral images (e.g., TDS).
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
            
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        loss = 'sparse_categorical_crossentropy' if num_classes > 2 else 'binary_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
             X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
             epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        Train the classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for deep learning)
            y_val: Validation labels (for deep learning)
            epochs: Number of epochs (for deep learning)
            batch_size: Batch size (for deep learning)
            
        Returns:
            Training history dictionary
        """
        history = {}
        
        if self.model_type in ['random_forest', 'svm']:
            # Traditional ML training
            self.model.fit(X_train, y_train)
            self.is_fitted = True
            
            # Store feature importances for Random Forest
            if self.model_type == 'random_forest':
                history['feature_importances'] = self.model.feature_importances_
        
        elif self.model_type == 'lstm':
            # LSTM training
            num_classes = len(np.unique(y_train))
            input_shape = (X_train.shape[1], X_train.shape[2])
            
            self.model = self.build_lstm_model(input_shape, num_classes)
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Train model
            keras_history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            history = keras_history.history
            self.is_fitted = True
        
        elif self.model_type == 'cnn':
            # CNN training
            num_classes = len(np.unique(y_train))
            input_shape = X_train.shape[1:]  # Exclude batch dimension
            
            self.model = self.build_cnn_model(input_shape, num_classes)
            
            # Prepare validation data
            validation_data = None
            if X_val is not None and y_val is not None:
                validation_data = (X_val, y_val)
            
            # Train model
            keras_history = self.model.fit(
                X_train, y_train,
                validation_data=validation_data,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            history = keras_history.history
            self.is_fitted = True
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type in ['random_forest', 'svm']:
            # Scale features
            X_scaled = self.scaler.transform(X)
            predictions = self.model.predict(X_scaled)
        else:
            # Deep learning prediction
            predictions = self.model.predict(X)
            if len(predictions.shape) > 1:
                predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        if self.model_type in ['random_forest', 'svm']:
            # Scale features
            X_scaled = self.scaler.transform(X)
            probabilities = self.model.predict_proba(X_scaled)
        else:
            # Deep learning prediction
            probabilities = self.model.predict(X)
            if len(probabilities.shape) == 1:
                # Binary classification - add complement probabilities
                probabilities = np.column_stack([1 - probabilities, probabilities])
        
        return probabilities
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the model performance.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Labels
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        if self.model_type not in ['random_forest', 'svm']:
            raise ValueError("Cross-validation only supported for traditional ML models")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Perform cross-validation
        scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='accuracy')
        f1_scores = cross_val_score(self.model, X_scaled, y_encoded, cv=cv, scoring='f1_weighted')
        
        return {
            'accuracy_scores': scores,
            'accuracy_mean': np.mean(scores),
            'accuracy_std': np.std(scores),
            'f1_scores': f1_scores,
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores)
        }
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                           param_grid: Dict = None) -> Dict:
        """
        Tune hyperparameters using grid search.
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Parameter grid for search
            
        Returns:
            Best parameters and scores
        """
        if self.model_type not in ['random_forest', 'svm']:
            raise ValueError("Hyperparameter tuning only supported for traditional ML models")
        
        # Default parameter grids
        if param_grid is None:
            if self.model_type == 'random_forest':
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                    'kernel': ['rbf', 'poly', 'sigmoid']
                }
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=5, 
            scoring='f1_weighted', n_jobs=-1
        )
        grid_search.fit(X_scaled, y_encoded)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        if self.model_type in ['random_forest', 'svm']:
            # Save traditional ML model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, filepath)
        else:
            # Save deep learning model
            self.model.save(filepath)
            
            # Save additional data
            metadata = {
                'scaler': self.scaler,
                'label_encoder': self.label_encoder,
                'model_type': self.model_type,
                'feature_names': self.feature_names
            }
            with open(filepath.replace('.h5', '_metadata.pkl'), 'wb') as f:
                pickle.dump(metadata, f)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        if self.model_type in ['random_forest', 'svm']:
            # Load traditional ML model
            model_data = joblib.load(filepath)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.model_type = model_data['model_type']
            self.feature_names = model_data.get('feature_names')
        else:
            # Load deep learning model
            self.model = keras.models.load_model(filepath)
            
            # Load additional data
            with open(filepath.replace('.h5', '_metadata.pkl'), 'rb') as f:
                metadata = pickle.load(f)
            self.scaler = metadata['scaler']
            self.label_encoder = metadata['label_encoder']
            self.model_type = metadata['model_type']
            self.feature_names = metadata.get('feature_names')
        
        self.is_fitted = True


class DataAugmentation:
    """Data augmentation techniques for radar data."""
    
    @staticmethod
    def add_noise(data: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to data."""
        noise = noise_level * np.random.randn(*data.shape)
        return data + noise
    
    @staticmethod
    def time_shift(data: np.ndarray, max_shift: int = 10) -> np.ndarray:
        """Apply random time shift."""
        shift = np.random.randint(-max_shift, max_shift + 1)
        if shift > 0:
            return np.concatenate([np.zeros((data.shape[0], shift)), data[:, :-shift]], axis=1)
        elif shift < 0:
            return np.concatenate([data[:, -shift:], np.zeros((data.shape[0], -shift))], axis=1)
        else:
            return data
    
    @staticmethod
    def scale_amplitude(data: np.ndarray, scale_range: Tuple[float, float] = (0.8, 1.2)) -> np.ndarray:
        """Apply random amplitude scaling."""
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        return data * scale_factor
    
    @staticmethod
    def augment_dataset(data: np.ndarray, labels: np.ndarray, 
                       augmentation_factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment dataset using various techniques.
        
        Args:
            data: Original data
            labels: Original labels
            augmentation_factor: How many times to augment each sample
            
        Returns:
            Augmented data and labels
        """
        augmented_data = [data]
        augmented_labels = [labels]
        
        for _ in range(augmentation_factor):
            # Apply random augmentations
            aug_data = data.copy()
            
            # Random noise
            if np.random.rand() > 0.5:
                aug_data = DataAugmentation.add_noise(aug_data, 0.05)
            
            # Random time shift
            if np.random.rand() > 0.5:
                aug_data = DataAugmentation.time_shift(aug_data, 5)
            
            # Random amplitude scaling
            if np.random.rand() > 0.5:
                aug_data = DataAugmentation.scale_amplitude(aug_data, (0.9, 1.1))
            
            augmented_data.append(aug_data)
            augmented_labels.append(labels)
        
        return np.concatenate(augmented_data, axis=0), np.concatenate(augmented_labels, axis=0)


if __name__ == "__main__":
    # Example usage
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    # Create synthetic feature matrix
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)  # Binary classification
    
    # Test Random Forest classifier
    classifier = RadarClassifier('random_forest')
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(X, y)
    
    # Train model
    history = classifier.train(X_train, y_train)
    
    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    
    print("Classifier example complete!")