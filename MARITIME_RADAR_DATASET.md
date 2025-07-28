# Maritime Radar Dataset Generator

A comprehensive system for generating large-scale realistic maritime radar datasets with labeled tracks for sea clutter and vessel targets. This dataset is designed for machine learning applications in radar-based maritime surveillance and target classification.

## 🎯 Overview

This system generates synthetic maritime radar datasets that include:

- **Sea clutter tracks**: Physically-based clutter using K-distribution and Weibull models
- **Vessel target tracks**: Realistic vessel motion patterns with various ship types
- **Multiple sea states**: Conditions ranging from calm (1) to very rough (7+)
- **Sequential detections**: Multi-point tracks rather than isolated detections
- **Complete metadata**: Comprehensive labeling for supervised learning

## 📊 Dataset Specifications

### Required Fields (Per Detection)

| Field | Type | Unit | Description |
|-------|------|------|-------------|
| `track_id` | String | - | Unique track identifier |
| `timestamp` | String | ISO-8601 UTC | Detection timestamp with ms resolution |
| `range_m` | Float | meters | Range from radar |
| `azimuth_deg` | Float | degrees | Azimuth angle (0-360°) |
| `elevation_deg` | Float | degrees | Elevation angle |
| `doppler_ms` | Float | m/s | Doppler velocity |
| `rcs_dbsm` | Float | dBsm | Radar cross-section |
| `snr_db` | Float | dB | Signal-to-noise ratio |
| `is_target` | Boolean | - | True for vessel, False for clutter |
| `sea_state` | Integer | 0-9 | Sea state scale |

### Dataset Characteristics

- **Size**: Configurable (1GB+ supported)
- **Tracks**: Thousands to millions of tracks
- **Detections per track**: 50-500 sequential points
- **Target/Clutter ratio**: ~20% targets, 80% clutter
- **Sea states**: Multiple conditions (calm to very rough)
- **Vessel types**: Small boats, fishing vessels, cargo ships, patrol boats

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd maritime-radar-dataset

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Demo Dataset

```bash
# Generate small demo dataset (~50MB)
python generate_maritime_dataset.py --quick_demo --analyze --export_ml
```

### 3. Generate Large Dataset

```bash
# Generate 2GB dataset with full analysis
python generate_maritime_dataset.py --size 2.0 --analyze --export_ml --output_dir my_dataset
```

### 4. Use Pre-built Scripts

```python
# Quick generation in Python
from src.maritime_radar_dataset import generate_large_maritime_dataset

dataset_file = generate_large_maritime_dataset(
    output_dir="maritime_data",
    target_size_gb=1.0
)
```

## 🔧 Advanced Usage

### Custom Radar Parameters

```python
from src.maritime_radar_dataset import RadarParameters, MaritimeRadarDatasetGenerator

# Custom radar configuration
radar_params = RadarParameters(
    frequency_hz=9.4e9,      # X-band
    bandwidth_hz=50e6,       # 50 MHz
    prf_hz=1000,            # 1 kHz PRF
    max_range_m=50000,      # 50 km max range
    antenna_gain_db=35.0    # 35 dB gain
)

# Generate with custom parameters
generator = MaritimeRadarDatasetGenerator(
    radar_params=radar_params,
    output_dir="custom_dataset"
)

dataset_file = generator.generate_dataset(
    n_clutter_tracks=5000,
    n_vessel_tracks=1000,
    sea_states=[2, 3, 4, 5],
    min_detections_per_track=100,
    max_detections_per_track=300
)
```

### Environmental Conditions

```python
from src.maritime_radar_dataset import EnvironmentalParameters

# Specific sea conditions
env_params = EnvironmentalParameters(
    sea_state=4,              # Moderate-rough
    wind_speed_ms=12.0,       # 12 m/s wind
    wave_height_m=2.5,        # 2.5m significant wave height
    temperature_c=20.0,       # 20°C
    humidity_percent=80.0     # 80% humidity
)
```

## 📈 Data Analysis

### Load and Analyze Dataset

```python
from src.dataset_processor import load_and_analyze_dataset

# Load and create visualizations
processor = load_and_analyze_dataset("dataset/maritime_radar_dataset.h5")

# Get detailed statistics
summary = processor.get_dataset_summary()
print(f"Total detections: {summary['basic_stats']['total_detections']:,}")
print(f"Target ratio: {summary['basic_stats']['target_ratio']:.3f}")
```

### Create Visualizations

```python
# Overview dashboard
overview_fig = processor.visualize_dataset_overview()
overview_fig.write_html("dataset_overview.html")

# Track analysis
track_fig = processor.visualize_track_analysis()
track_fig.write_html("track_analysis.html")
```

### Export for Machine Learning

```python
# Prepare ML-ready datasets
ml_files = processor.export_for_training(
    output_dir="ml_data",
    feature_type='detection',  # or 'track'
    normalize=True
)

# Files created:
# - detection_train.h5
# - detection_val.h5  
# - detection_test.h5
# - detection_scaler.pkl
# - detection_feature_info.json
```

## 🤖 Machine Learning Integration

### Loading ML Data

```python
import pandas as pd
import json

# Load training data
train_df = pd.read_hdf('ml_data/detection_train.h5', key='data')

# Load feature information
with open('ml_data/detection_feature_info.json', 'r') as f:
    feature_info = json.load(f)

# Separate features and labels
feature_cols = [col for col in train_df.columns if col != 'is_target']
X_train = train_df[feature_cols]
y_train = train_df['is_target']
```

### Example Training Pipeline

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
val_df = pd.read_hdf('ml_data/detection_val.h5', key='data')
X_val = val_df[feature_cols]
y_val = val_df['is_target']

predictions = rf.predict(X_val)
print(classification_report(y_val, predictions))
```

## 🏗️ System Architecture

### Core Components

1. **Physical Models**
   - `SeaClutterModel`: K-distribution and Weibull clutter generation
   - `VesselTargetModel`: Realistic vessel motion and RCS modeling

2. **Dataset Generation**
   - `MaritimeRadarDatasetGenerator`: Main generation orchestrator
   - Multi-processing support for large datasets
   - Configurable parameters and sea states

3. **Data Processing**
   - `MaritimeRadarDatasetProcessor`: Analysis and ML preparation
   - Feature engineering and normalization
   - Train/validation/test splitting

4. **Visualization**
   - Interactive Plotly dashboards
   - Track trajectory analysis
   - Statistical distributions

### File Structure

```
maritime_radar_dataset/
├── maritime_radar_dataset.h5      # Main dataset (HDF5)
├── maritime_radar_dataset.parquet # Compressed version
├── maritime_radar_sample.csv      # Sample for inspection
├── metadata.json                  # Dataset metadata
├── maritime_radar_train.h5        # Training split
├── maritime_radar_val.h5          # Validation split
├── maritime_radar_test.h5         # Test split
├── dataset_overview.html          # Analysis dashboard
├── track_analysis.html            # Track visualization
└── ml_ready/                      # ML-prepared data
    ├── detection_train.h5
    ├── detection_val.h5
    ├── detection_test.h5
    ├── detection_scaler.pkl
    ├── detection_feature_info.json
    └── usage_example.py
```

## 🔬 Physical Modeling

### Sea Clutter Models

#### K-Distribution Model
The K-distribution is widely used for modeling sea clutter returns:

- **Shape parameter**: Varies with sea state (5.0 for calm, 0.4 for hurricane)
- **Temporal correlation**: Moving average filtering for realistic correlation
- **Range dependence**: Empirical models based on real data

#### Weibull Distribution Model
Alternative clutter model with different characteristics:

- **Shape/scale parameters**: Tuned for each sea state
- **Phase randomization**: Uniform random phase
- **Amplitude correlation**: Spatial and temporal coherence

### Vessel Target Models

#### Motion Patterns
Realistic vessel trajectories based on vessel type:

- **Small boats**: High maneuverability, frequent course changes
- **Fishing vessels**: Moderate speed, occasional direction changes
- **Cargo ships**: Steady course, minimal maneuvering
- **Patrol boats**: High speed, tactical maneuvers

#### RCS Modeling
Radar cross-section varies with:

- **Vessel type**: Size-dependent base RCS
- **Aspect angle**: Bow/stern vs. beam aspects
- **Fluctuation**: Swerling models for realistic variation

## 📐 Performance Specifications

### Generation Performance

| Dataset Size | Tracks | Generation Time* | Memory Usage |
|--------------|--------|------------------|--------------|
| 100 MB | ~2,500 | 30 seconds | ~1 GB |
| 1 GB | ~25,000 | 5 minutes | ~4 GB |
| 5 GB | ~125,000 | 25 minutes | ~8 GB |
| 10 GB | ~250,000 | 50 minutes | ~12 GB |

*Using multiprocessing on 8-core system

### Data Quality Metrics

- **Track continuity**: >95% tracks have continuous motion
- **Physical realism**: RCS and Doppler within expected ranges
- **Statistical distribution**: Matches real-world radar data characteristics
- **Class balance**: Configurable target/clutter ratios

## 🎛️ Configuration Options

### Command Line Interface

```bash
python generate_maritime_dataset.py \
    --size 2.0 \                    # Dataset size in GB
    --output_dir my_dataset \       # Output directory
    --analyze \                     # Generate analysis
    --export_ml \                   # Prepare ML data
    --quick_demo                    # Small demo dataset
```

### Programmatic Configuration

```python
# Radar system parameters
radar_params = RadarParameters(
    frequency_hz=9.4e9,         # Operating frequency
    bandwidth_hz=50e6,          # Signal bandwidth
    pulse_width_s=1e-6,         # Pulse width
    prf_hz=1000,               # Pulse repetition frequency
    antenna_gain_db=35.0,       # Antenna gain
    noise_figure_db=3.0,        # Receiver noise figure
    max_range_m=50000          # Maximum detection range
)

# Environmental parameters
env_params = EnvironmentalParameters(
    sea_state=3,               # Sea state (0-9)
    wind_speed_ms=10.0,        # Wind speed
    wave_height_m=2.0,         # Significant wave height
    temperature_c=15.0,        # Air temperature
    humidity_percent=75.0      # Relative humidity
)

# Generation parameters
generator.generate_dataset(
    n_clutter_tracks=10000,    # Number of clutter tracks
    n_vessel_tracks=2000,      # Number of vessel tracks
    sea_states=[1,3,5,7],      # Sea states to include
    min_detections_per_track=50,  # Minimum track length
    max_detections_per_track=500, # Maximum track length
    use_multiprocessing=True   # Enable parallel processing
)
```

## 🧪 Validation and Testing

### Data Quality Checks

1. **Statistical validation**: Compare generated data distributions with real radar data
2. **Physical consistency**: Verify RCS, Doppler, and range relationships
3. **Track continuity**: Ensure smooth vessel trajectories
4. **Temporal coherence**: Check timestamp sequences and detection rates

### Performance Benchmarks

Run built-in validation:

```python
from src.maritime_radar_dataset import validate_dataset

# Validate generated dataset
validation_results = validate_dataset("dataset/maritime_radar_dataset.h5")
print(f"Validation score: {validation_results['overall_score']:.3f}")
```

## 🔄 Integration with Existing Systems

### Export Formats

- **HDF5**: Efficient for large datasets, supports compression
- **Parquet**: Excellent compression, compatible with Apache Spark
- **CSV**: Human-readable, compatible with all tools
- **JSON**: Metadata and configuration files

### API Integration

```python
# Streaming interface for real-time applications
from src.maritime_radar_dataset import RadarDataStream

stream = RadarDataStream(radar_params, env_params)
for detection in stream.generate_detections():
    # Process detection in real-time
    process_radar_detection(detection)
```

## 🚨 Troubleshooting

### Common Issues

**Memory errors with large datasets:**
```bash
# Reduce batch size or use streaming generation
python generate_maritime_dataset.py --size 10.0 --no_multiprocessing
```

**Slow generation:**
```bash
# Enable multiprocessing (default) and check CPU usage
htop  # Monitor CPU utilization during generation
```

**Installation issues:**
```bash
# Install with conda for complex dependencies
conda install -c conda-forge numpy scipy pandas plotly h5py
pip install -r requirements.txt
```

### Debug Mode

```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Generate with debugging
generator = MaritimeRadarDatasetGenerator(output_dir="debug_dataset")
generator.generate_dataset(n_clutter_tracks=10, n_vessel_tracks=2)
```

## 📚 References and Citations

### Radar Modeling References

1. Nathanson, F.E. "Radar Design Principles" - Sea clutter models
2. Ward, K.D. "Sea Clutter: Scattering, the K-Distribution and Radar Performance"
3. Skolnik, M.I. "Introduction to Radar Systems" - General radar principles

### Maritime Surveillance Applications

1. Greidanus, H. "Satellite-based vessel detection" - Target characteristics
2. Brusch, S. "Ship surveillance with TerraSAR-X" - Maritime radar applications

### Statistical Models

1. Jakeman, E. "K-distribution modeling of sea clutter"
2. Watts, S. "Radar detection prediction in sea clutter using the compound K-distribution"

## 🤝 Contributing

### Development Setup

```bash
# Development installation
git clone <repository-url>
cd maritime-radar-dataset
pip install -e .

# Run tests
python -m pytest tests/

# Code style
black src/
flake8 src/
```

### Adding New Features

1. **New clutter models**: Extend `SeaClutterModel` class
2. **Vessel types**: Add to `VesselTargetModel.vessel_types`
3. **Environmental effects**: Modify `EnvironmentalParameters`
4. **Export formats**: Extend `_save_dataset` method

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 📞 Support

For questions, issues, or contributions:

- **Issues**: Create GitHub issues for bugs or feature requests
- **Discussions**: Use GitHub discussions for general questions
- **Email**: Contact the development team

---

**Generated datasets are for research and development purposes. Validate against real data before operational use.**