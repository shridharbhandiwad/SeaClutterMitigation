"""
Visualization module for radar clutter classification.
Includes plots for Doppler spectra, TDS images, classification results, and performance metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, List, Tuple, Optional, Union
import pandas as pd


class RadarVisualizer:
    """
    Visualization class for radar data and classification results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_iq_time_series(self, iq_data: np.ndarray, 
                           sampling_frequency: float = 1000.0,
                           title: str = "I/Q Time Series") -> plt.Figure:
        """
        Plot I/Q time series data.
        
        Args:
            iq_data: Complex I/Q data
            sampling_frequency: Sampling frequency in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 1, figsize=self.figsize)
        
        # Time axis
        t = np.arange(len(iq_data)) / sampling_frequency
        
        # Plot I and Q components
        axes[0].plot(t, np.real(iq_data), 'b-', label='I (In-phase)', alpha=0.7)
        axes[0].plot(t, np.imag(iq_data), 'r-', label='Q (Quadrature)', alpha=0.7)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(title)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot magnitude
        magnitude = np.abs(iq_data)
        axes[1].plot(t, magnitude, 'g-', label='Magnitude')
        axes[1].set_ylabel('Magnitude')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot phase
        phase = np.angle(iq_data)
        axes[2].plot(t, phase, 'm-', label='Phase')
        axes[2].set_ylabel('Phase (rad)')
        axes[2].set_xlabel('Time (s)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_doppler_spectrum(self, spectrum: np.ndarray, 
                            frequencies: np.ndarray,
                            title: str = "Doppler Spectrum",
                            log_scale: bool = True) -> plt.Figure:
        """
        Plot Doppler spectrum.
        
        Args:
            spectrum: Magnitude spectrum
            frequencies: Frequency array
            title: Plot title
            log_scale: Whether to use logarithmic scale
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if log_scale:
            spectrum_plot = 20 * np.log10(spectrum + 1e-10)
            ylabel = 'Magnitude (dB)'
        else:
            spectrum_plot = spectrum
            ylabel = 'Magnitude'
        
        ax.plot(frequencies, spectrum_plot, 'b-', linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Highlight peak frequency
        peak_idx = np.argmax(spectrum)
        peak_freq = frequencies[peak_idx]
        peak_mag = spectrum_plot[peak_idx]
        ax.plot(peak_freq, peak_mag, 'ro', markersize=8, label=f'Peak: {peak_freq:.1f} Hz')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_time_doppler_spectrum(self, tds: np.ndarray, 
                                 times: np.ndarray, 
                                 frequencies: np.ndarray,
                                 title: str = "Time-Doppler Spectrum") -> plt.Figure:
        """
        Plot Time-Doppler Spectrum (TDS) as a 2D image.
        
        Args:
            tds: TDS magnitude data
            times: Time array
            frequencies: Frequency array
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Convert to dB scale
        tds_db = 20 * np.log10(tds + 1e-10)
        
        # Create mesh plot
        im = ax.imshow(tds_db, aspect='auto', origin='lower', 
                      extent=[times[0], times[-1], frequencies[0], frequencies[-1]],
                      cmap='viridis')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title(title)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude (dB)')
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_tds(self, tds: np.ndarray, 
                           times: np.ndarray, 
                           frequencies: np.ndarray,
                           title: str = "Interactive Time-Doppler Spectrum"):
        """
        Create interactive Time-Doppler Spectrum plot using Plotly.
        
        Args:
            tds: TDS magnitude data
            times: Time array
            frequencies: Frequency array
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Convert to dB scale
        tds_db = 20 * np.log10(tds + 1e-10)
        
        fig = go.Figure(data=go.Heatmap(
            z=tds_db,
            x=times,
            y=frequencies,
            colorscale='Viridis',
            colorbar=dict(title="Magnitude (dB)")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_3d_spectrum(self, tds: np.ndarray, 
                        times: np.ndarray, 
                        frequencies: np.ndarray,
                        title: str = "3D Time-Doppler Spectrum") -> plt.Figure:
        """
        Create 3D surface plot of TDS.
        
        Args:
            tds: TDS magnitude data
            times: Time array
            frequencies: Frequency array
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create meshgrid
        T, F = np.meshgrid(times, frequencies)
        
        # Convert to dB scale
        tds_db = 20 * np.log10(tds + 1e-10)
        
        # Create surface plot
        surf = ax.plot_surface(T, F, tds_db, cmap='viridis', alpha=0.8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_zlabel('Magnitude (dB)')
        ax.set_title(title)
        
        # Add colorbar
        fig.colorbar(surf, ax=ax, shrink=0.5)
        
        return fig
    
    def plot_feature_distributions(self, features: np.ndarray, 
                                 labels: np.ndarray,
                                 feature_names: List[str] = None,
                                 max_features: int = 12) -> plt.Figure:
        """
        Plot feature distributions for different classes.
        
        Args:
            features: Feature matrix
            labels: Class labels
            feature_names: Names of features
            max_features: Maximum number of features to plot
            
        Returns:
            Matplotlib figure
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(features.shape[1])]
        
        # Select subset of features if too many
        n_features = min(features.shape[1], max_features)
        selected_features = features[:, :n_features]
        selected_names = feature_names[:n_features]
        
        # Create subplots
        n_rows = int(np.ceil(np.sqrt(n_features)))
        n_cols = int(np.ceil(n_features / n_rows))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, (feature_idx, feature_name) in enumerate(zip(range(n_features), selected_names)):
            ax = axes[i]
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                data = selected_features[mask, feature_idx]
                
                ax.hist(data, bins=30, alpha=0.6, color=color, 
                       label=f'Class {label}', density=True)
            
            ax.set_title(feature_name)
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, importances: np.ndarray, 
                              feature_names: List[str] = None,
                              top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance from tree-based models.
        
        Args:
            importances: Feature importance values
            feature_names: Names of features
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        sorted_importances = importances[indices]
        sorted_names = [feature_names[i] for i in indices]
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.3)))
        
        bars = ax.barh(range(len(sorted_importances)), sorted_importances)
        ax.set_yticks(range(len(sorted_importances)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3)
        
        # Color bars by importance
        for bar, importance in zip(bars, sorted_importances):
            bar.set_color(plt.cm.viridis(importance / np.max(sorted_importances)))
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, 
                            class_names: List[str] = None,
                            normalize: bool = True) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: Names of classes
            normalize: Whether to normalize the matrix
            
        Returns:
            Matplotlib figure
        """
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, 
                      y_scores: np.ndarray,
                      title: str = "ROC Curve") -> plt.Figure:
        """
        Plot ROC curve.
        
        Args:
            y_true: True binary labels
            y_scores: Target scores (probability estimates)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        auc = np.trapz(tpr, fpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, 
                                   y_scores: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True binary labels
            y_scores: Target scores (probability estimates)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        auc_pr = np.trapz(precision, recall)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, 'b-', linewidth=2, 
               label=f'PR Curve (AUC = {auc_pr:.3f})')
        
        # Baseline (random classifier)
        baseline = np.sum(y_true) / len(y_true)
        ax.axhline(y=baseline, color='r', linestyle='--', 
                  label=f'Random Classifier (AP = {baseline:.3f})')
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_learning_curves(self, history: Dict,
                           title: str = "Learning Curves") -> plt.Figure:
        """
        Plot training and validation learning curves.
        
        Args:
            history: Training history dictionary
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        if 'loss' in history:
            epochs = range(1, len(history['loss']) + 1)
            axes[0].plot(epochs, history['loss'], 'b-', label='Training Loss')
            if 'val_loss' in history:
                axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Model Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in history:
            epochs = range(1, len(history['accuracy']) + 1)
            axes[1].plot(epochs, history['accuracy'], 'b-', label='Training Accuracy')
            if 'val_accuracy' in history:
                axes[1].plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(title)
        plt.tight_layout()
        return fig
    
    def plot_feature_space_2d(self, features: np.ndarray, 
                             labels: np.ndarray,
                             method: str = 'pca',
                             title: str = "2D Feature Space Visualization") -> plt.Figure:
        """
        Plot 2D visualization of feature space using dimensionality reduction.
        
        Args:
            features: Feature matrix
            labels: Class labels
            method: Dimensionality reduction method ('pca' or 'tsne')
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if method == 'pca':
            reducer = PCA(n_components=2)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'pca' or 'tsne'")
        
        # Apply dimensionality reduction
        features_2d = reducer.fit_transform(features)
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                      c=[color], label=f'Class {label}', alpha=0.6, s=30)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add explained variance for PCA
        if method == 'pca':
            var_explained = reducer.explained_variance_ratio_
            ax.text(0.02, 0.98, f'Explained Variance: {var_explained.sum():.3f}',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_classification_report_heatmap(self, y_true: np.ndarray, 
                                         y_pred: np.ndarray,
                                         class_names: List[str] = None) -> plt.Figure:
        """
        Plot classification report as a heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            
        Returns:
            Matplotlib figure
        """
        from sklearn.metrics import classification_report
        
        if class_names is None:
            class_names = [f'Class {i}' for i in np.unique(y_true)]
        
        # Get classification report as dictionary
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Convert to DataFrame for visualization
        df = pd.DataFrame(report).iloc[:-1, :-3].T  # Exclude support and averages
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(df, annot=True, cmap='Blues', fmt='.3f', ax=ax)
        
        ax.set_title('Classification Report')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Classes')
        
        plt.tight_layout()
        return fig
    
    def create_dashboard(self, radar_data: Dict, 
                        classification_results: Dict) -> go.Figure:
        """
        Create an interactive dashboard with multiple plots.
        
        Args:
            radar_data: Dictionary containing radar data
            classification_results: Dictionary containing classification results
            
        Returns:
            Plotly figure with subplots
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Doppler Spectrum', 'Time-Doppler Spectrum', 
                          'ROC Curve', 'Feature Importance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Doppler spectrum
        if 'spectrum' in radar_data and 'frequencies' in radar_data:
            spectrum_db = 20 * np.log10(radar_data['spectrum'] + 1e-10)
            fig.add_trace(
                go.Scatter(x=radar_data['frequencies'], y=spectrum_db, 
                          mode='lines', name='Doppler Spectrum'),
                row=1, col=1
            )
        
        # TDS heatmap
        if 'tds' in radar_data:
            tds_db = 20 * np.log10(radar_data['tds'] + 1e-10)
            fig.add_trace(
                go.Heatmap(z=tds_db, colorscale='Viridis', showscale=False),
                row=1, col=2
            )
        
        # ROC curve
        if 'fpr' in classification_results and 'tpr' in classification_results:
            fig.add_trace(
                go.Scatter(x=classification_results['fpr'], 
                          y=classification_results['tpr'],
                          mode='lines', name='ROC Curve'),
                row=2, col=1
            )
            # Diagonal line
            fig.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode='lines', 
                          line=dict(dash='dash'), name='Random'),
                row=2, col=1
            )
        
        # Feature importance
        if 'feature_importance' in classification_results:
            importance = classification_results['feature_importance']
            feature_names = classification_results.get('feature_names', 
                                                     [f'F{i}' for i in range(len(importance))])
            
            # Top 10 features
            top_indices = np.argsort(importance)[-10:]
            fig.add_trace(
                go.Bar(x=importance[top_indices], 
                      y=[feature_names[i] for i in top_indices],
                      orientation='h'),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Radar Classification Dashboard")
        
        return fig


if __name__ == "__main__":
    # Example usage
    
    # Create synthetic data for demonstration
    fs = 1000
    t = np.arange(2048) / fs
    
    # Synthetic I/Q signal
    iq_signal = np.exp(1j * 2 * np.pi * 50 * t) + 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    
    # Synthetic spectrum
    frequencies = np.linspace(-fs/2, fs/2, 1024)
    spectrum = np.exp(-((frequencies - 50)**2) / (2 * 10**2)) + 0.1 * np.random.randn(1024)
    
    # Create visualizer
    visualizer = RadarVisualizer()
    
    # Plot I/Q time series
    fig1 = visualizer.plot_iq_time_series(iq_signal, fs)
    plt.show()
    
    # Plot Doppler spectrum
    fig2 = visualizer.plot_doppler_spectrum(spectrum, frequencies)
    plt.show()
    
    print("Visualization example complete!")