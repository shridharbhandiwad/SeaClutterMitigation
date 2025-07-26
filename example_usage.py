#!/usr/bin/env python3
"""
Example usage of the Radar Sea Clutter Classification system.

This script demonstrates various components of the system with simple examples.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append('src')

from data_loader import RadarDataLoader
from signal_processing import RadarSignalProcessor, create_complex_iq_from_real
from feature_extraction import RadarFeatureExtractor
from classifier import RadarClassifier
from visualization import RadarVisualizer


def example_1_data_loading():
    """Example 1: Data loading and preprocessing."""
    print("=" * 50)
    print("EXAMPLE 1: DATA LOADING AND PREPROCESSING")
    print("=" * 50)
    
    # Initialize data loader
    loader = RadarDataLoader(data_dir="data/")
    
    # Generate synthetic data (since we don't have real datasets)
    synthetic_data = loader._generate_synthetic_data('demo')
    
    # Preprocess the data
    processed_data = loader.preprocess_data(synthetic_data)
    
    print(f"Original data shape: {synthetic_data['I'].shape}")
    print(f"Processed data shape: {processed_data['I'].shape}")
    print(f"Number of range cells: {processed_data['metadata']['range_cells']}")
    print(f"Time samples per cell: {processed_data['metadata']['time_samples']}")
    print(f"Labels: {processed_data['metadata']['labels']}")
    
    return processed_data


def example_2_signal_processing(data):
    """Example 2: Signal processing and spectral analysis."""
    print("\n" + "=" * 50)
    print("EXAMPLE 2: SIGNAL PROCESSING")
    print("=" * 50)
    
    # Create complex I/Q data
    iq_data = create_complex_iq_from_real(data['I'], data['Q'])
    
    # Initialize signal processor
    fs = data['metadata']['sampling_frequency']
    processor = RadarSignalProcessor(fs)
    
    # Take first range cell for demonstration
    cell_iq = iq_data[0, :]
    
    # Segment the time series
    segmented_data, indices = processor.segment_time_series(
        cell_iq, window_size=512, overlap_ratio=0.5
    )
    print(f"Original signal length: {len(cell_iq)}")
    print(f"Number of segments: {segmented_data.shape[1]}")
    print(f"Segment length: {segmented_data.shape[2]}")
    
    # Compute Doppler spectrum for first segment
    segment = segmented_data[0, 0, :]
    spectrum, frequencies = processor.compute_doppler_spectrum(segment)
    print(f"Spectrum length: {len(spectrum)}")
    print(f"Frequency range: {frequencies[0]:.1f} to {frequencies[-1]:.1f} Hz")
    
    # Compute Time-Doppler Spectrum
    tds, times, freqs = processor.compute_time_doppler_spectrum(cell_iq)
    print(f"TDS shape: {tds.shape}")
    print(f"Time samples: {len(times)}, Frequency bins: {len(freqs)}")
    
    return {
        'iq_data': iq_data,
        'spectrum': spectrum,
        'frequencies': frequencies,
        'tds': tds,
        'times': times,
        'tds_frequencies': freqs,
        'sampling_frequency': fs
    }


def example_3_feature_extraction(signal_data):
    """Example 3: Feature extraction."""
    print("\n" + "=" * 50)
    print("EXAMPLE 3: FEATURE EXTRACTION")
    print("=" * 50)
    
    # Initialize feature extractor
    extractor = RadarFeatureExtractor(signal_data['sampling_frequency'])
    
    # Extract features from first range cell
    cell_iq = signal_data['iq_data'][0, :]
    
    # Extract all features
    features = extractor.extract_all_features(
        cell_iq, 
        signal_data['spectrum'], 
        signal_data['frequencies'],
        signal_data['tds']
    )
    
    print(f"Total features extracted: {len(features)}")
    print("\nSample features:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name}: {value:.4f}")
    
    # Get feature names
    feature_names = extractor.get_feature_names()
    print(f"\nTotal available feature types: {len(feature_names)}")
    
    return features, feature_names


def example_4_classification(data, signal_data):
    """Example 4: Classification."""
    print("\n" + "=" * 50)
    print("EXAMPLE 4: CLASSIFICATION")
    print("=" * 50)
    
    # Prepare feature matrix for all range cells
    extractor = RadarFeatureExtractor(signal_data['sampling_frequency'])
    
    all_features = []
    labels = data['metadata']['labels']
    
    print("Extracting features from all range cells...")
    for i in range(data['I'].shape[0]):
        cell_iq = signal_data['iq_data'][i, :]
        
        # Compute spectrum for this cell
        processor = RadarSignalProcessor(signal_data['sampling_frequency'])
        spectrum, frequencies = processor.compute_doppler_spectrum(cell_iq)
        
        # Extract features (without TDS for speed)
        features = extractor.extract_all_features(cell_iq, spectrum, frequencies)
        all_features.append(list(features.values()))
    
    # Convert to numpy array
    X = np.array(all_features)
    y = labels
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Unique classes: {np.unique(y)}")
    
    # Test different classifiers
    models = ['random_forest', 'svm']
    results = {}
    
    for model_type in models:
        print(f"\nTraining {model_type}...")
        
        # Initialize and train classifier
        classifier = RadarClassifier(model_type)
        X_train, X_test, y_train, y_test = classifier.prepare_data(X, y, test_size=0.3)
        
        # Train
        history = classifier.train(X_train, y_train)
        
        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)
        
        results[model_type] = {
            'classifier': classifier,
            'metrics': metrics,
            'history': history
        }
        
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
        if 'roc_auc' in metrics:
            print(f"  ROC AUC: {metrics['roc_auc']:.3f}")
    
    return results


def example_5_visualization(data, signal_data, classification_results):
    """Example 5: Visualization."""
    print("\n" + "=" * 50)
    print("EXAMPLE 5: VISUALIZATION")
    print("=" * 50)
    
    # Initialize visualizer
    visualizer = RadarVisualizer()
    
    # Create output directory
    output_dir = "example_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating visualizations...")
    
    # 1. I/Q Time Series
    fig1 = visualizer.plot_iq_time_series(
        signal_data['iq_data'][0, :], 
        signal_data['sampling_frequency'],
        "Example I/Q Time Series"
    )
    fig1.savefig(f"{output_dir}/iq_example.png", dpi=150, bbox_inches='tight')
    plt.close(fig1)
    
    # 2. Doppler Spectrum
    fig2 = visualizer.plot_doppler_spectrum(
        signal_data['spectrum'], 
        signal_data['frequencies'],
        "Example Doppler Spectrum"
    )
    fig2.savefig(f"{output_dir}/spectrum_example.png", dpi=150, bbox_inches='tight')
    plt.close(fig2)
    
    # 3. Time-Doppler Spectrum
    fig3 = visualizer.plot_time_doppler_spectrum(
        signal_data['tds'], 
        signal_data['times'], 
        signal_data['tds_frequencies'],
        "Example Time-Doppler Spectrum"
    )
    fig3.savefig(f"{output_dir}/tds_example.png", dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    # 4. Confusion Matrix
    best_model = 'random_forest'  # Choose best performing model
    cm = classification_results[best_model]['metrics']['confusion_matrix']
    fig4 = visualizer.plot_confusion_matrix(cm, ['Clutter', 'Target'])
    fig4.savefig(f"{output_dir}/confusion_matrix_example.png", dpi=150, bbox_inches='tight')
    plt.close(fig4)
    
    # 5. Feature Importance (if available)
    if 'feature_importances' in classification_results[best_model]['history']:
        importances = classification_results[best_model]['history']['feature_importances']
        fig5 = visualizer.plot_feature_importance(importances, top_n=15)
        fig5.savefig(f"{output_dir}/feature_importance_example.png", dpi=150, bbox_inches='tight')
        plt.close(fig5)
    
    print(f"Visualizations saved to {output_dir}/")


def example_6_advanced_features():
    """Example 6: Advanced features demonstration."""
    print("\n" + "=" * 50)
    print("EXAMPLE 6: ADVANCED FEATURES")
    print("=" * 50)
    
    # Data augmentation example
    print("Data Augmentation Example:")
    original_data = np.random.randn(100, 1000)  # 100 samples, 1000 features each
    original_labels = np.random.randint(0, 2, 100)
    
    from classifier import DataAugmentation
    augmented_data, augmented_labels = DataAugmentation.augment_dataset(
        original_data, original_labels, augmentation_factor=2
    )
    
    print(f"Original data shape: {original_data.shape}")
    print(f"Augmented data shape: {augmented_data.shape}")
    print(f"Augmentation factor: {augmented_data.shape[0] / original_data.shape[0]:.1f}x")
    
    # Cross-validation example
    print("\nCross-Validation Example:")
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=20, n_classes=2, random_state=42)
    
    classifier = RadarClassifier('random_forest')
    cv_results = classifier.cross_validate(X, y, cv=5)
    
    print(f"Cross-validation accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
    print(f"Cross-validation F1: {cv_results['f1_mean']:.3f} ± {cv_results['f1_std']:.3f}")
    
    # Hyperparameter tuning example
    print("\nHyperparameter Tuning Example:")
    tuning_results = classifier.tune_hyperparameters(X, y)
    print(f"Best parameters: {tuning_results['best_params']}")
    print(f"Best score: {tuning_results['best_score']:.3f}")


def main():
    """Run all examples."""
    print("RADAR SEA CLUTTER CLASSIFICATION - EXAMPLE USAGE")
    print("=" * 60)
    
    # Example 1: Data loading
    data = example_1_data_loading()
    
    # Example 2: Signal processing
    signal_data = example_2_signal_processing(data)
    
    # Example 3: Feature extraction
    features, feature_names = example_3_feature_extraction(signal_data)
    
    # Example 4: Classification
    classification_results = example_4_classification(data, signal_data)
    
    # Example 5: Visualization
    example_5_visualization(data, signal_data, classification_results)
    
    # Example 6: Advanced features
    example_6_advanced_features()
    
    print("\n" + "=" * 60)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run the main pipeline: python main.py --dataset synthetic --model random_forest --visualize")
    print("2. Try different models: --model svm, --model lstm, --model cnn")
    print("3. Experiment with parameters: --window_size 512 --overlap 0.75")
    print("4. Use real datasets when available")


if __name__ == "__main__":
    main()