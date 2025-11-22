"""
Advanced Neural Network Utilities for Fraud Detection
===============================================

This module provides comprehensive utilities for building, training, and evaluating
neural networks with advanced features like focal loss, custom metrics, and 
hyperparameter optimization.

Author: [Your Name]
Course: STW7088CEM - Artificial Neural Network
Date: November 2024
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict, List, Tuple, Optional, Union
import optuna


class FocalLoss(keras.losses.Loss):
    """
    Focal Loss implementation for handling class imbalance.
    
    Formula: FL(p_t) = -α_t(1-p_t)^γ log(p_t)
    
    Args:
        alpha: Weighting factor for rare class (default: 0.25)
        gamma: Focusing parameter for hard examples (default: 2.0)
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        # Ensure predictions are probabilities
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
        
        # Calculate focal loss
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * tf.pow(1 - p_t, self.gamma) * tf.math.log(p_t)
        
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'gamma': self.gamma
        })
        return config


class WeightedBinaryCrossentropy(keras.losses.Loss):
    """
    Weighted binary crossentropy for imbalanced datasets.
    """
    
    def __init__(self, pos_weight=1.0, name='weighted_bce'):
        super().__init__(name=name)
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-8, 1 - 1e-8)
        
        # Weighted binary crossentropy
        loss = -(self.pos_weight * y_true * tf.math.log(y_pred) + 
                (1 - y_true) * tf.math.log(1 - y_pred))
        
        return tf.reduce_mean(loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({'pos_weight': self.pos_weight})
        return config


class F1Score(keras.metrics.Metric):
    """
    F1 Score metric implementation for Keras.
    """
    
    def __init__(self, threshold=0.5, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1
    
    def reset_state(self):
        self.true_positives.assign(0)
        self.false_positives.assign(0)
        self.false_negatives.assign(0)


def create_advanced_fraud_model(input_dim: int, 
                               architecture: str = 'standard',
                               dropout_rate: float = 0.3,
                               l2_reg: float = 0.001,
                               batch_norm: bool = True) -> keras.Model:
    """
    Create advanced neural network models for fraud detection.
    
    Args:
        input_dim: Number of input features
        architecture: Model architecture type ('standard', 'deep', 'wide', 'residual')
        dropout_rate: Dropout rate for regularization
        l2_reg: L2 regularization strength
        batch_norm: Whether to use batch normalization
    
    Returns:
        Compiled Keras model
    """
    
    # Input layer
    inputs = layers.Input(shape=(input_dim,), name='input_features')
    
    # L2 regularization
    regularizer = keras.regularizers.l2(l2_reg) if l2_reg > 0 else None
    
    if architecture == 'standard':
        # Standard feedforward architecture
        x = layers.Dense(128, kernel_regularizer=regularizer)(inputs)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(64, kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(32, kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        
    elif architecture == 'deep':
        # Deep architecture with more layers
        layer_sizes = [256, 128, 64, 32, 16]
        x = inputs
        
        for i, size in enumerate(layer_sizes):
            x = layers.Dense(size, kernel_regularizer=regularizer)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate * (0.8 ** i))(x)  # Decreasing dropout
            
    elif architecture == 'wide':
        # Wide architecture with larger layers
        x = layers.Dense(512, kernel_regularizer=regularizer)(inputs)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(256, kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        x = layers.Dense(128, kernel_regularizer=regularizer)(x)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout_rate / 2)(x)
        
    elif architecture == 'residual':
        # Residual connections for deeper networks
        def residual_block(x, units, dropout_rate):
            residual = x
            
            # First layer
            x = layers.Dense(units, kernel_regularizer=regularizer)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Dropout(dropout_rate)(x)
            
            # Second layer
            x = layers.Dense(units, kernel_regularizer=regularizer)(x)
            if batch_norm:
                x = layers.BatchNormalization()(x)
            
            # Add residual connection if dimensions match
            if residual.shape[-1] == units:
                x = layers.Add()([x, residual])
            
            x = layers.Activation('relu')(x)
            return x
        
        x = layers.Dense(128, kernel_regularizer=regularizer)(inputs)
        if batch_norm:
            x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        
        x = residual_block(x, 128, dropout_rate)
        x = residual_block(x, 64, dropout_rate)
        x = layers.Dropout(dropout_rate / 2)(x)
        
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    # Output layer
    outputs = layers.Dense(1, activation='sigmoid', name='fraud_probability')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name=f'{architecture}_fraud_detector')
    
    return model


def create_bayesian_optimizer(X_train: np.ndarray, 
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            n_trials: int = 100) -> optuna.Study:
    """
    Create Bayesian optimization study for hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of optimization trials
    
    Returns:
        Optuna study object with optimization results
    """
    
    def objective(trial):
        # Hyperparameter suggestions
        architecture = trial.suggest_categorical('architecture', 
                                                ['standard', 'deep', 'wide', 'residual'])
        dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.6)
        l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
        epochs = trial.suggest_int('epochs', 20, 100)
        
        # Loss function choice
        loss_type = trial.suggest_categorical('loss_type', ['focal', 'weighted_bce', 'bce'])
        
        # Create model
        model = create_advanced_fraud_model(
            input_dim=X_train.shape[1],
            architecture=architecture,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg,
            batch_norm=True
        )
        
        # Choose loss function
        if loss_type == 'focal':
            alpha = trial.suggest_float('focal_alpha', 0.1, 0.5)
            gamma = trial.suggest_float('focal_gamma', 1.0, 3.0)
            loss = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_type == 'weighted_bce':
            pos_weight = trial.suggest_float('pos_weight', 1.0, 100.0)
            loss = WeightedBinaryCrossentropy(pos_weight=pos_weight)
        else:
            loss = 'binary_crossentropy'
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['accuracy', F1Score()]
        )
        
        # Early stopping and pruning
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        pruning_callback = TFKerasPruningCallback(trial, 'val_loss')
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, pruning_callback],
            verbose=0
        )
        
        # Get validation predictions
        y_val_pred = model.predict(X_val, verbose=0)
        val_auc = roc_auc_score(y_val, y_val_pred)
        
        return val_auc
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10)
    )
    
    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def plot_training_history(history: keras.callbacks.History,
                         model_name: str = "Neural Network") -> None:
    """
    Plot comprehensive training history.
    
    Args:
        history: Keras training history
        model_name: Name of the model for plot titles
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training and validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', alpha=0.8)
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', alpha=0.8)
    axes[0, 0].set_title(f'{model_name} - Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training and validation accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', alpha=0.8)
    if 'val_accuracy' in history.history:
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', alpha=0.8)
    axes[0, 1].set_title(f'{model_name} - Accuracy')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # F1 Score (if available)
    if 'f1_score' in history.history:
        axes[1, 0].plot(history.history['f1_score'], label='Training F1', alpha=0.8)
        if 'val_f1_score' in history.history:
            axes[1, 0].plot(history.history['val_f1_score'], label='Validation F1', alpha=0.8)
        axes[1, 0].set_title(f'{model_name} - F1 Score')
        axes[1, 0].set_xlabel('Epochs')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate', alpha=0.8)
        axes[1, 1].set_title(f'{model_name} - Learning Rate')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        # Plot loss difference if learning rate not available
        train_loss = np.array(history.history['loss'])
        val_loss = np.array(history.history.get('val_loss', train_loss))
        loss_diff = val_loss - train_loss
        axes[1, 1].plot(loss_diff, label='Val Loss - Train Loss', alpha=0.8, color='red')
        axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 1].set_title(f'{model_name} - Overfitting Indicator')
        axes[1, 1].set_xlabel('Epochs')
        axes[1, 1].set_ylabel('Loss Difference')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_model_comprehensive(model: keras.Model,
                                X_test: np.ndarray,
                                y_test: np.ndarray,
                                model_name: str = "Model") -> Dict[str, float]:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        model_name: Name of the model
    
    Returns:
        Dictionary of evaluation metrics
    """
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'AUC': roc_auc_score(y_test, y_pred_proba),
        'F1': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Accuracy': np.mean(y_test == y_pred)
    }
    
    # Calculate optimal threshold using Youden's J statistic
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions with optimal threshold
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    metrics['Optimal_Threshold'] = optimal_threshold
    metrics['Optimal_F1'] = f1_score(y_test, y_pred_optimal)
    metrics['Optimal_Precision'] = precision_score(y_test, y_pred_optimal)
    metrics['Optimal_Recall'] = recall_score(y_test, y_pred_optimal)
    
    print(f"=== {model_name} Evaluation Results ===")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return metrics


class ModelInterpretability:
    """
    Model interpretability utilities using SHAP and LIME.
    """
    
    def __init__(self, model: keras.Model, X_background: np.ndarray):
        self.model = model
        self.X_background = X_background
        
    def explain_with_shap(self, X_explain: np.ndarray, 
                         max_evals: int = 100) -> 'shap.Explanation':
        """
        Generate SHAP explanations for model predictions.
        """
        try:
            import shap
            # Create explainer
            explainer = shap.KernelExplainer(
                self.model.predict, 
                self.X_background[:100]  # Use subset for efficiency
            )
            
            # Generate explanations
            shap_values = explainer.shap_values(X_explain[:min(50, len(X_explain))], 
                                              nsamples=max_evals)
            
            return shap_values
        except ImportError:
            print("SHAP not installed. Please install with: pip install shap")
            return None
    
    def explain_with_lime(self, X_explain: np.ndarray, 
                         feature_names: List[str] = None) -> List:
        """
        Generate LIME explanations for model predictions.
        """
        try:
            import lime.lime_tabular
            
            # Create LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_background,
                feature_names=feature_names,
                class_names=['Normal', 'Fraud'],
                mode='classification'
            )
            
            explanations = []
            for i in range(min(5, len(X_explain))):  # Explain first 5 instances
                exp = explainer.explain_instance(
                    X_explain[i], 
                    lambda x: self.model.predict(x),
                    num_features=10
                )
                explanations.append(exp)
            
            return explanations
        except ImportError:
            print("LIME not installed. Please install with: pip install lime")
            return []


def create_model_comparison_report(models: Dict[str, Dict], 
                                 save_path: Optional[str] = None) -> None:
    """
    Create comprehensive model comparison report.
    
    Args:
        models: Dictionary of model results
        save_path: Optional path to save the report
    """
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, results in models.items():
        if 'metrics' in results:
            comparison_data.append({
                'Model': model_name,
                **results['metrics']
            })
    
    import pandas as pd
    df_comparison = pd.DataFrame(comparison_data)
    
    # Display comparison
    print("=== MODEL COMPARISON REPORT ===")
    print(df_comparison.to_string(index=False, float_format='%.4f'))
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_to_plot = ['AUC', 'F1', 'Precision', 'Recall']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]
        if metric in df_comparison.columns:
            bars = ax.bar(df_comparison['Model'], df_comparison[metric])
            ax.set_title(f'Model Comparison - {metric}')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                       f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Save report if path provided
    if save_path:
        df_comparison.to_csv(save_path, index=False)
        print(f"Report saved to: {save_path}")
    
    return df_comparison