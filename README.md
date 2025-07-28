# Maritime Radar Dataset Generator & Sea Clutter Classifier

A comprehensive Python toolkit for generating large-scale maritime radar datasets and performing radar-based sea clutter classification. This project implements both synthetic dataset generation with physically-based models and complete machine learning pipelines for distinguishing between sea clutter and vessel targets.

## 🆕 NEW: Maritime Radar Dataset Generator

Generate massive, realistic maritime radar datasets with labeled tracks:

- **Physical Models**: K-distribution and Weibull sea clutter models
- **Realistic Targets**: Vessel motion patterns with multiple ship types
- **Large Scale**: Generate 1GB+ datasets with millions of detections
- **Track Data**: Sequential detections forming continuous tracks
- **ML Ready**: Automatic train/validation/test splits and feature preparation

## Features

### 🎯 Core Capabilities
- **Dataset Generation**: Create large-scale synthetic maritime radar datasets
- **Data Loading**: Support for IPIX and CSIR radar datasets with synthetic data generation
- **Signal Processing**: Complete I/Q signal processing pipeline with windowing and FFT
- **Feature Extraction**: 80+ spectral, temporal, and image-based features
- **Classification**: Multiple ML/DL models (Random Forest, SVM, LSTM, CNN)
- **Visualization**: Comprehensive plotting and interactive dashboards

### 📊 Signal Processing
- I/Q time series segmentation with overlapping windows
- Doppler spectrum computation using FFT
- Time-Doppler Spectrum (TDS) analysis using STFT
- Power spectral density estimation
- Clutter removal and preprocessing

### 🔍 Feature Extraction
- **Spectral Features**: Centroid, spread, entropy, bandwidth, peaks
- **Temporal Features**: Statistical moments, zero-crossing rate, autocorrelation
- **Image Features**: Local Binary Patterns (LBP), Gabor filters
- **Complexity Measures**: Fractal dimension, approximate/sample entropy
- **STFT Features**: Short-time spectral analysis

### 🤖 Machine Learning Models
- **Traditional ML**: Random Forest, Support Vector Machines
- **Deep Learning**: LSTM networks, Convolutional Neural Networks
- **Model Evaluation**: ROC curves, precision-recall, confusion matrices
- **Hyperparameter Tuning**: Grid search with cross-validation

### 📈 Visualization
- I/Q time series plots
- Doppler spectrum visualization
- Time-Doppler Spectrum heatmaps and 3D plots
- Feature distribution analysis
- Classification performance metrics
- Interactive Plotly dashboards

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone <repository-url>
cd radar-clutter-classifier

# Install dependencies
pip install -r requirements.txt
```

### Dependencies
The project requires the following main packages:
- `numpy`, `scipy`: Numerical computing and signal processing
- `scikit-learn`: Machine learning algorithms
- `tensorflow`: Deep learning models
- `matplotlib`, `seaborn`, `plotly`: Visualization
- `h5py`: Data file handling
- `scikit-image`: Image processing features

## Quick Start

### 1. Generate Maritime Radar Dataset
```bash
# Quick demo dataset
python3 test_dataset_generation.py

# Large-scale dataset (requires dependencies)
python generate_maritime_dataset.py --size 2.0 --analyze --export_ml
```

### 2. Run Example Usage
```bash
python example_usage.py
```
This demonstrates all major components with synthetic data.

### 2. Complete Pipeline
```bash
# Basic usage with synthetic data
python main.py --dataset synthetic --model random_forest --visualize

# Advanced usage
python main.py --dataset synthetic --model svm --window_size 512 --overlap 0.75 --visualize --save_model models/svm_model.pkl
```

### 3. Try Different Models
```bash
# Random Forest
python main.py --model random_forest --visualize

# Support Vector Machine
python main.py --model svm --visualize

# LSTM Neural Network
python main.py --model lstm --visualize

# Convolutional Neural Network
python main.py --model cnn --visualize
```

## Project Structure

```
radar-clutter-classifier/
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── signal_processing.py    # I/Q signal processing
│   ├── feature_extraction.py   # Feature computation
│   ├── classifier.py           # ML/DL models
│   └── visualization.py        # Plotting and visualization
├── main.py                     # Main pipeline script
├── example_usage.py            # Example demonstrations
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Usage Guide

### Data Loading
```python
from src.data_loader import RadarDataLoader, load_radar_data

# Load synthetic data
data = load_radar_data('synthetic')

# Load real datasets (when available)
data = load_radar_data('ipix')  # or 'csir'
```

### Signal Processing
```python
from src.signal_processing import RadarSignalProcessor, create_complex_iq_from_real

# Create complex I/Q data
iq_data = create_complex_iq_from_real(data['I'], data['Q'])

# Initialize processor
processor = RadarSignalProcessor(sampling_frequency=1000)

# Segment time series
segments, indices = processor.segment_time_series(
    iq_data[0], window_size=1024, overlap_ratio=0.5
)

# Compute Doppler spectrum
spectrum, frequencies = processor.compute_doppler_spectrum(segments[0, 0])

# Compute Time-Doppler Spectrum
tds, times, freqs = processor.compute_time_doppler_spectrum(iq_data[0])
```

### Feature Extraction
```python
from src.feature_extraction import RadarFeatureExtractor

extractor = RadarFeatureExtractor(sampling_frequency=1000)

# Extract all features
features = extractor.extract_all_features(
    iq_data[0],      # I/Q time series
    spectrum,        # Doppler spectrum
    frequencies,     # Frequency array
    tds             # Time-Doppler Spectrum
)

print(f"Extracted {len(features)} features")
```

### Classification
```python
from src.classifier import RadarClassifier

# Initialize classifier
classifier = RadarClassifier('random_forest')

# Prepare data
X_train, X_test, y_train, y_test = classifier.prepare_data(features, labels)

# Train model
history = classifier.train(X_train, y_train)

# Evaluate
metrics = classifier.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.3f}")
```

### Visualization
```python
from src.visualization import RadarVisualizer

visualizer = RadarVisualizer()

# Plot I/Q time series
fig1 = visualizer.plot_iq_time_series(iq_data[0], sampling_frequency=1000)

# Plot Doppler spectrum
fig2 = visualizer.plot_doppler_spectrum(spectrum, frequencies)

# Plot Time-Doppler Spectrum
fig3 = visualizer.plot_time_doppler_spectrum(tds, times, frequencies)

# Plot ROC curve
fig4 = visualizer.plot_roc_curve(y_true, y_scores)
```

## Command Line Options

### Main Pipeline (`main.py`)
```bash
Options:
  --dataset {synthetic,ipix,csir}    Dataset to use (default: synthetic)
  --model {random_forest,svm,lstm,cnn}  Model type (default: random_forest)
  --window_size INT                  Window size for segmentation (default: 1024)
  --overlap FLOAT                   Overlap ratio (default: 0.5)
  --augment                         Apply data augmentation
  --visualize                       Generate visualizations
  --save_model PATH                 Save trained model
  --load_model PATH                 Load pre-trained model
  --output_dir PATH                 Output directory (default: results)
```

## Datasets

### 🌊 Maritime Radar Dataset (NEW)
Generate large-scale realistic maritime radar datasets with:

**Dataset Specifications:**
- **Fields**: TrackID, Range, Azimuth, Elevation, Doppler, RCS, SNR, Timestamp, Labels
- **Size**: 1GB+ (configurable, millions of detections)
- **Tracks**: Sequential detections (50-500 points per track)
- **Targets**: Vessels (small boats, fishing vessels, cargo ships, patrol boats)
- **Clutter**: Physically-based sea clutter (K-distribution, Weibull models)
- **Conditions**: Multiple sea states (calm to very rough)

**Quick Generation:**
```bash
# Simple test (no dependencies required)
python3 test_dataset_generation.py

# Large dataset with full features
python generate_maritime_dataset.py --size 2.0 --analyze --export_ml --output_dir my_dataset
```

**Output Files:**
- `maritime_radar_dataset.h5` - Main dataset (HDF5 format)
- `maritime_radar_dataset.parquet` - Compressed version
- `maritime_radar_train/val/test.h5` - ML-ready splits
- `dataset_overview.html` - Interactive analysis dashboard
- `metadata.json` - Complete dataset documentation

### Synthetic Data (Original)
The original system generates realistic synthetic radar data for demonstration:
- 20 range cells (15 clutter, 5 targets)
- 8192 time samples per cell
- Configurable sea states and target characteristics
- Amplitude modulation for sea clutter
- Higher frequency coherent targets

### Real Datasets (Template Support)
- **IPIX Dataset**: McMaster University radar data
- **CSIR Dataset**: Council for Scientific and Industrial Research data

*Note: Real dataset loaders are provided as templates. Update URLs and file formats as needed.*

## Performance Metrics

The system evaluates classifiers using:
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **Precision-Recall**: Precision vs recall curves
- **Confusion Matrix**: Detailed classification breakdown

## Advanced Features

### Data Augmentation
```python
from src.classifier import DataAugmentation

# Augment dataset
augmented_data, augmented_labels = DataAugmentation.augment_dataset(
    original_data, original_labels, augmentation_factor=2
)
```

### Cross-Validation
```python
# Perform 5-fold cross-validation
cv_results = classifier.cross_validate(features, labels, cv=5)
print(f"CV Accuracy: {cv_results['accuracy_mean']:.3f} ± {cv_results['accuracy_std']:.3f}")
```

### Hyperparameter Tuning
```python
# Automatic hyperparameter optimization
tuning_results = classifier.tune_hyperparameters(X_train, y_train)
print(f"Best parameters: {tuning_results['best_params']}")
```

## Research Applications

This toolkit supports research in:
- **Radar Signal Processing**: Advanced I/Q analysis techniques
- **Feature Engineering**: Novel feature extraction for radar data
- **Machine Learning**: Comparative analysis of ML/DL approaches
- **Maritime Surveillance**: Small vessel detection in sea clutter
- **Sensor Fusion**: Integration with other sensor modalities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use this toolkit in your research, please cite:
```bibtex
@software{radar_clutter_classifier,
  title={Radar Sea Clutter Classification Toolkit},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/radar-clutter-classifier}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- McMaster University for the IPIX dataset
- CSIR for the small boat and clutter dataset
- The radar signal processing community for valuable insights

## Support

For questions and support:
- Create an issue on GitHub
- Check the example usage script
- Review the comprehensive documentation in each module

---

**Happy radar processing! 🎯📡**