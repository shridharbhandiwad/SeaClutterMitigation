#!/usr/bin/env python3
"""
Main pipeline for radar sea clutter classification.

This script demonstrates the complete workflow:
1. Load and preprocess radar data
2. Extract features from I/Q time series
3. Train classifiers (Random Forest, SVM, LSTM, CNN)
4. Evaluate performance and visualize results

Usage:
    python main.py --dataset synthetic --model random_forest --visualize
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_loader import RadarDataLoader, load_radar_data
from signal_processing import RadarSignalProcessor, create_complex_iq_from_real
from feature_extraction import RadarFeatureExtractor
from classifier import RadarClassifier, DataAugmentation
from visualization import RadarVisualizer


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Radar Sea Clutter Classification Pipeline')
    
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'ipix', 'csir'],
                       help='Dataset to use')
    
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'lstm', 'cnn'],
                       help='Model type to use')
    
    parser.add_argument('--window_size', type=int, default=1024,
                       help='Window size for segmentation')
    
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Overlap ratio for windowing')
    
    parser.add_argument('--augment', action='store_true',
                       help='Apply data augmentation')
    
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations')
    
    parser.add_argument('--save_model', type=str, default=None,
                       help='Path to save trained model')
    
    parser.add_argument('--load_model', type=str, default=None,
                       help='Path to load pre-trained model')
    
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    
    return parser.parse_args()


def load_and_preprocess_data(dataset_name: str, data_dir: str = "data/"):
    """
    Load and preprocess radar data.
    
    Args:
        dataset_name: Name of dataset to load
        data_dir: Data directory
        
    Returns:
        Dictionary containing processed radar data
    """
    print(f"Loading {dataset_name} dataset...")
    
    # Load data
    data = load_radar_data(dataset_name, data_dir)
    
    # Create complex I/Q data
    iq_data = create_complex_iq_from_real(data['I'], data['Q'])
    
    # Get metadata
    fs = data['metadata'].get('sampling_frequency', 1000.0)
    labels = data['metadata'].get('labels')
    
    print(f"Data shape: {iq_data.shape}")
    print(f"Sampling frequency: {fs} Hz")
    if labels is not None:
        print(f"Labels: {len(labels)} samples, {len(np.unique(labels))} classes")
    
    return {
        'iq_data': iq_data,
        'labels': labels,
        'sampling_frequency': fs,
        'metadata': data['metadata']
    }


def process_signals_and_extract_features(iq_data: np.ndarray, 
                                       sampling_frequency: float,
                                       window_size: int = 1024,
                                       overlap_ratio: float = 0.5):
    """
    Process signals and extract features.
    
    Args:
        iq_data: Complex I/Q data
        sampling_frequency: Sampling frequency
        window_size: Window size for segmentation
        overlap_ratio: Overlap ratio
        
    Returns:
        Dictionary containing features and intermediate results
    """
    print("Processing signals and extracting features...")
    
    # Initialize processors
    signal_processor = RadarSignalProcessor(sampling_frequency)
    feature_extractor = RadarFeatureExtractor(sampling_frequency)
    
    n_range_cells = iq_data.shape[0]
    all_features = []
    all_spectra = []
    all_tds = []
    
    for range_cell in range(n_range_cells):
        print(f"Processing range cell {range_cell + 1}/{n_range_cells}")
        
        # Get I/Q data for this range cell
        cell_iq = iq_data[range_cell, :]
        
        # Segment the time series
        segmented_data, window_indices = signal_processor.segment_time_series(
            cell_iq, window_size, overlap_ratio
        )
        
        # Process each segment
        for segment_idx in range(segmented_data.shape[1]):
            segment_iq = segmented_data[0, segment_idx, :]  # Remove extra dimension
            
            # Compute Doppler spectrum
            spectrum, frequencies = signal_processor.compute_doppler_spectrum(segment_iq)
            all_spectra.append(spectrum)
            
            # Compute Time-Doppler Spectrum
            tds, times, freqs = signal_processor.compute_time_doppler_spectrum(segment_iq)
            all_tds.append(tds)
            
            # Extract features
            features = feature_extractor.extract_all_features(
                segment_iq, spectrum, frequencies, tds
            )
            all_features.append(features)
    
    # Convert features to numpy array
    feature_names = list(all_features[0].keys())
    feature_matrix = np.array([[feat[name] for name in feature_names] 
                              for feat in all_features])
    
    print(f"Extracted {len(feature_names)} features from {len(all_features)} segments")
    
    return {
        'features': feature_matrix,
        'feature_names': feature_names,
        'spectra': np.array(all_spectra),
        'tds_images': all_tds,
        'frequencies': frequencies,
        'times': times
    }


def prepare_labels_for_segments(original_labels: np.ndarray, 
                               n_range_cells: int, 
                               segments_per_cell: int):
    """
    Prepare labels for segmented data.
    
    Args:
        original_labels: Original labels per range cell
        n_range_cells: Number of range cells
        segments_per_cell: Number of segments per range cell
        
    Returns:
        Labels for all segments
    """
    segment_labels = []
    
    for range_cell in range(n_range_cells):
        cell_label = original_labels[range_cell]
        # Replicate label for all segments from this range cell
        segment_labels.extend([cell_label] * segments_per_cell)
    
    return np.array(segment_labels)


def train_and_evaluate_model(features: np.ndarray, 
                           labels: np.ndarray,
                           model_type: str,
                           apply_augmentation: bool = False):
    """
    Train and evaluate classifier.
    
    Args:
        features: Feature matrix
        labels: Labels
        model_type: Type of model to train
        apply_augmentation: Whether to apply data augmentation
        
    Returns:
        Dictionary containing trained model and evaluation results
    """
    print(f"Training {model_type} classifier...")
    
    # Initialize classifier
    classifier = RadarClassifier(model_type)
    
    # Apply data augmentation if requested
    if apply_augmentation and model_type in ['lstm', 'cnn']:
        print("Applying data augmentation...")
        features, labels = DataAugmentation.augment_dataset(features, labels, 
                                                           augmentation_factor=1)
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(features, labels)
    
    # Reshape data for deep learning models
    if model_type == 'lstm':
        # Reshape for LSTM: (samples, timesteps, features)
        # For simplicity, treat features as sequence
        n_timesteps = min(50, X_train.shape[1])  # Limit timesteps
        n_features = X_train.shape[1] // n_timesteps
        
        if X_train.shape[1] % n_timesteps != 0:
            # Pad or truncate to make divisible
            new_size = n_timesteps * n_features
            X_train = X_train[:, :new_size]
            X_test = X_test[:, :new_size]
        
        X_train = X_train.reshape(X_train.shape[0], n_timesteps, n_features)
        X_test = X_test.reshape(X_test.shape[0], n_timesteps, n_features)
    
    elif model_type == 'cnn':
        # Reshape for CNN: (samples, height, width, channels)
        # Create 2D representation of features
        height = int(np.sqrt(X_train.shape[1]))
        width = X_train.shape[1] // height
        
        if height * width != X_train.shape[1]:
            # Pad to make square
            new_size = height * height
            if new_size > X_train.shape[1]:
                # Pad with zeros
                padding = new_size - X_train.shape[1]
                X_train = np.pad(X_train, ((0, 0), (0, padding)), mode='constant')
                X_test = np.pad(X_test, ((0, 0), (0, padding)), mode='constant')
                width = height
        
        X_train = X_train.reshape(X_train.shape[0], height, width, 1)
        X_test = X_test.reshape(X_test.shape[0], height, width, 1)
    
    # Train model
    if model_type in ['lstm', 'cnn']:
        # Split training data for validation
        val_split = 0.2
        val_size = int(len(X_train) * val_split)
        
        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_train = X_train[val_size:]
        y_train = y_train[val_size:]
        
        history = classifier.train(X_train, y_train, X_val, y_val, epochs=50)
    else:
        history = classifier.train(X_train, y_train)
    
    # Evaluate model
    metrics = classifier.evaluate(X_test, y_test)
    
    # Get predictions for ROC/PR curves
    y_pred_proba = classifier.predict_proba(X_test)
    
    # Calculate ROC curve
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
    else:
        fpr, tpr, precision, recall = None, None, None, None
    
    print(f"Model Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1 Score: {metrics['f1_score']:.3f}")
    if 'roc_auc' in metrics:
        print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    
    return {
        'classifier': classifier,
        'metrics': metrics,
        'history': history,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'precision': precision,
        'recall': recall
    }


def create_visualizations(radar_data: dict, 
                        feature_data: dict, 
                        classification_results: dict,
                        output_dir: str):
    """
    Create and save visualizations.
    
    Args:
        radar_data: Original radar data
        feature_data: Processed features and spectra
        classification_results: Classification results
        output_dir: Output directory for saving plots
    """
    print("Creating visualizations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = RadarVisualizer()
    
    # 1. Plot I/Q time series (first range cell)
    if 'iq_data' in radar_data:
        fig1 = visualizer.plot_iq_time_series(
            radar_data['iq_data'][0, :], 
            radar_data['sampling_frequency'],
            "I/Q Time Series - Range Cell 1"
        )
        fig1.savefig(os.path.join(output_dir, 'iq_time_series.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
    
    # 2. Plot Doppler spectrum (first segment)
    if 'spectra' in feature_data and 'frequencies' in feature_data:
        fig2 = visualizer.plot_doppler_spectrum(
            feature_data['spectra'][0], 
            feature_data['frequencies'],
            "Doppler Spectrum - First Segment"
        )
        fig2.savefig(os.path.join(output_dir, 'doppler_spectrum.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # 3. Plot Time-Doppler Spectrum
    if 'tds_images' in feature_data and len(feature_data['tds_images']) > 0:
        tds = feature_data['tds_images'][0]
        times = feature_data['times']
        frequencies = feature_data['frequencies']
        
        fig3 = visualizer.plot_time_doppler_spectrum(
            tds, times, frequencies,
            "Time-Doppler Spectrum - First Segment"
        )
        fig3.savefig(os.path.join(output_dir, 'tds_spectrum.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
    
    # 4. Plot feature distributions
    if 'features' in feature_data:
        # Get labels for all segments
        n_segments = len(feature_data['features'])
        n_range_cells = radar_data['iq_data'].shape[0]
        segments_per_cell = n_segments // n_range_cells
        
        segment_labels = prepare_labels_for_segments(
            radar_data['labels'], n_range_cells, segments_per_cell
        )
        
        fig4 = visualizer.plot_feature_distributions(
            feature_data['features'], segment_labels, 
            feature_data['feature_names'][:12]  # Show first 12 features
        )
        fig4.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close(fig4)
    
    # 5. Plot feature importance (for tree-based models)
    if 'history' in classification_results and 'feature_importances' in classification_results['history']:
        fig5 = visualizer.plot_feature_importance(
            classification_results['history']['feature_importances'],
            feature_data['feature_names']
        )
        fig5.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close(fig5)
    
    # 6. Plot confusion matrix
    cm = classification_results['metrics']['confusion_matrix']
    fig6 = visualizer.plot_confusion_matrix(cm, ['Clutter', 'Target'])
    fig6.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close(fig6)
    
    # 7. Plot ROC curve
    if classification_results['fpr'] is not None:
        fig7 = visualizer.plot_roc_curve(
            classification_results['y_test'], 
            classification_results['y_pred_proba'][:, 1]
        )
        fig7.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close(fig7)
    
    # 8. Plot Precision-Recall curve
    if classification_results['precision'] is not None:
        fig8 = visualizer.plot_precision_recall_curve(
            classification_results['y_test'], 
            classification_results['y_pred_proba'][:, 1]
        )
        fig8.savefig(os.path.join(output_dir, 'precision_recall_curve.png'), dpi=300, bbox_inches='tight')
        plt.close(fig8)
    
    # 9. Plot learning curves (for deep learning)
    if 'history' in classification_results and 'loss' in classification_results['history']:
        fig9 = visualizer.plot_learning_curves(classification_results['history'])
        fig9.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300, bbox_inches='tight')
        plt.close(fig9)
    
    # 10. Plot 2D feature space
    if 'features' in feature_data:
        fig10 = visualizer.plot_feature_space_2d(
            feature_data['features'], segment_labels, method='pca'
        )
        fig10.savefig(os.path.join(output_dir, 'feature_space_2d.png'), dpi=300, bbox_inches='tight')
        plt.close(fig10)
    
    print(f"Visualizations saved to {output_dir}")


def main():
    """Main pipeline function."""
    args = parse_arguments()
    
    print("=" * 60)
    print("RADAR SEA CLUTTER CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load and preprocess data
    radar_data = load_and_preprocess_data(args.dataset)
    
    # 2. Process signals and extract features
    feature_data = process_signals_and_extract_features(
        radar_data['iq_data'], 
        radar_data['sampling_frequency'],
        args.window_size,
        args.overlap
    )
    
    # 3. Prepare labels for segments
    n_segments = len(feature_data['features'])
    n_range_cells = radar_data['iq_data'].shape[0]
    segments_per_cell = n_segments // n_range_cells
    
    segment_labels = prepare_labels_for_segments(
        radar_data['labels'], n_range_cells, segments_per_cell
    )
    
    # 4. Train and evaluate classifier
    if args.load_model:
        print(f"Loading pre-trained model from {args.load_model}")
        classifier = RadarClassifier(args.model)
        classifier.load_model(args.load_model)
        
        # Evaluate on current data
        X_train, X_test, y_train, y_test = classifier.prepare_data(
            feature_data['features'], segment_labels
        )
        metrics = classifier.evaluate(X_test, y_test)
        y_pred_proba = classifier.predict_proba(X_test)
        
        classification_results = {
            'classifier': classifier,
            'metrics': metrics,
            'history': {},
            'X_test': X_test,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba,
            'fpr': None,
            'tpr': None,
            'precision': None,
            'recall': None
        }
        
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
            classification_results.update({
                'fpr': fpr, 'tpr': tpr, 
                'precision': precision, 'recall': recall
            })
    else:
        classification_results = train_and_evaluate_model(
            feature_data['features'], 
            segment_labels,
            args.model,
            args.augment
        )
    
    # 5. Save model if requested
    if args.save_model:
        classification_results['classifier'].save_model(args.save_model)
        print(f"Model saved to {args.save_model}")
    
    # 6. Create visualizations
    if args.visualize:
        create_visualizations(
            radar_data, 
            feature_data, 
            classification_results,
            args.output_dir
        )
    
    # 7. Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Number of samples: {len(segment_labels)}")
    print(f"Number of features: {feature_data['features'].shape[1]}")
    print(f"Accuracy: {classification_results['metrics']['accuracy']:.3f}")
    print(f"F1 Score: {classification_results['metrics']['f1_score']:.3f}")
    if 'roc_auc' in classification_results['metrics']:
        print(f"ROC AUC: {classification_results['metrics']['roc_auc']:.3f}")
    print(f"Results saved to: {args.output_dir}")
    
    print("\nClassification Report:")
    print(classification_results['metrics']['classification_report'])


if __name__ == "__main__":
    main()