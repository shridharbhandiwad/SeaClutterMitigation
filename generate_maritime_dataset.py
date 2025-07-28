#!/usr/bin/env python3
"""
Maritime Radar Dataset Generation Demo

Complete workflow for generating, analyzing, and preparing a large-scale
maritime radar dataset with labeled tracks for machine learning applications.

This script demonstrates:
1. Dataset generation with physical models
2. Data analysis and visualization
3. ML-ready data preparation
4. Train/validation/test splits

Usage:
    python generate_maritime_dataset.py --size 1.0 --output_dir maritime_data
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from maritime_radar_dataset import (
    MaritimeRadarDatasetGenerator, 
    generate_large_maritime_dataset,
    RadarParameters,
    EnvironmentalParameters
)
from dataset_processor import (
    MaritimeRadarDatasetProcessor,
    load_and_analyze_dataset
)


def main():
    parser = argparse.ArgumentParser(description="Generate Maritime Radar Dataset")
    parser.add_argument("--size", type=float, default=1.0, 
                       help="Target dataset size in GB (default: 1.0)")
    parser.add_argument("--output_dir", type=str, default="maritime_radar_dataset",
                       help="Output directory (default: maritime_radar_dataset)")
    parser.add_argument("--analyze", action="store_true",
                       help="Perform dataset analysis after generation")
    parser.add_argument("--no_multiprocessing", action="store_true",
                       help="Disable multiprocessing (useful for debugging)")
    parser.add_argument("--quick_demo", action="store_true",
                       help="Generate small demo dataset for testing")
    parser.add_argument("--export_ml", action="store_true",
                       help="Export ML-ready datasets")
    
    args = parser.parse_args()
    
    print("="*80)
    print("MARITIME RADAR DATASET GENERATOR")
    print("="*80)
    print(f"Target size: {args.size} GB")
    print(f"Output directory: {args.output_dir}")
    print(f"Multiprocessing: {not args.no_multiprocessing}")
    print()
    
    start_time = time.time()
    
    if args.quick_demo:
        print("🚀 Generating quick demo dataset...")
        # Small dataset for testing
        generator = MaritimeRadarDatasetGenerator(output_dir=args.output_dir)
        dataset_file = generator.generate_dataset(
            n_clutter_tracks=100,
            n_vessel_tracks=20,
            sea_states=[1, 3, 5],
            min_detections_per_track=50,
            max_detections_per_track=200,
            use_multiprocessing=not args.no_multiprocessing
        )
        
        # Create splits
        splits = generator.create_train_val_test_split()
        print(f"✅ Demo dataset created: {dataset_file}")
        
    else:
        print("🚀 Generating large-scale maritime radar dataset...")
        dataset_file = generate_large_maritime_dataset(
            output_dir=args.output_dir,
            target_size_gb=args.size
        )
        print(f"✅ Dataset created: {dataset_file}")
    
    generation_time = time.time() - start_time
    print(f"⏱️  Generation time: {generation_time:.1f} seconds")
    
    # Get actual file size
    if os.path.exists(dataset_file):
        file_size_gb = os.path.getsize(dataset_file) / (1024**3)
        print(f"📊 Actual dataset size: {file_size_gb:.2f} GB")
    
    # Analysis
    if args.analyze:
        print("\n" + "="*60)
        print("DATASET ANALYSIS")
        print("="*60)
        
        try:
            processor = load_and_analyze_dataset(dataset_file, create_visualizations=False)
            
            # Create and save visualizations
            print("\n📈 Creating visualizations...")
            
            # Overview visualization
            overview_fig = processor.visualize_dataset_overview()
            overview_path = os.path.join(args.output_dir, "dataset_overview.html")
            overview_fig.write_html(overview_path)
            print(f"📊 Overview saved: {overview_path}")
            
            # Track analysis
            track_fig = processor.visualize_track_analysis()
            track_path = os.path.join(args.output_dir, "track_analysis.html")
            track_fig.write_html(track_path)
            print(f"📈 Track analysis saved: {track_path}")
            
            # Print detailed statistics
            summary = processor.get_dataset_summary()
            print("\n📋 DETAILED STATISTICS:")
            print(f"   • Total detections: {summary['basic_stats']['total_detections']:,}")
            print(f"   • Unique tracks: {summary['basic_stats']['unique_tracks']:,}")
            print(f"   • Target ratio: {summary['basic_stats']['target_ratio']:.1%}")
            print(f"   • Sea states: {summary['basic_stats']['unique_sea_states']}")
            print(f"   • Mean track length: {summary['track_statistics']['mean_track_length']:.1f}")
            
            print(f"\n📐 FIELD RANGES:")
            for field, stats in summary['field_statistics'].items():
                if field == 'range_m':
                    print(f"   • {field}: {stats['min']/1000:.1f} - {stats['max']/1000:.1f} km")
                elif field in ['azimuth_deg', 'elevation_deg']:
                    print(f"   • {field}: {stats['min']:.1f} - {stats['max']:.1f}°")
                else:
                    print(f"   • {field}: {stats['min']:.1f} - {stats['max']:.1f}")
            
        except Exception as e:
            print(f"❌ Analysis failed: {e}")
    
    # Export ML-ready data
    if args.export_ml:
        print("\n" + "="*60)
        print("ML DATA PREPARATION")
        print("="*60)
        
        try:
            processor = MaritimeRadarDatasetProcessor(dataset_file)
            
            # Export detection-level features
            print("🔬 Preparing detection-level features...")
            ml_dir = os.path.join(args.output_dir, "ml_ready")
            detection_files = processor.export_for_training(
                output_dir=ml_dir,
                feature_type='detection',
                normalize=True
            )
            
            print("✅ Detection-level files:")
            for split, path in detection_files.items():
                if split in ['train', 'val', 'test']:
                    size_mb = os.path.getsize(path) / (1024**2)
                    print(f"   • {split}: {path} ({size_mb:.1f} MB)")
            
            # Export track-level features
            print("\n🎯 Preparing track-level features...")
            track_files = processor.export_for_training(
                output_dir=ml_dir,
                feature_type='track',
                normalize=True
            )
            
            print("✅ Track-level files:")
            for split, path in track_files.items():
                if split in ['train', 'val', 'test']:
                    size_mb = os.path.getsize(path) / (1024**2)
                    print(f"   • {split}: {path} ({size_mb:.1f} MB)")
            
            # Create usage example
            create_usage_example(ml_dir)
            
        except Exception as e:
            print(f"❌ ML preparation failed: {e}")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Total execution time: {total_time:.1f} seconds")
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"📁 Dataset location: {args.output_dir}")
    print(f"📊 Main file: {dataset_file}")
    
    if args.analyze:
        print(f"📈 Visualizations: {args.output_dir}/dataset_overview.html")
        print(f"                  {args.output_dir}/track_analysis.html")
    
    if args.export_ml:
        print(f"🤖 ML-ready data: {args.output_dir}/ml_ready/")
        print(f"📝 Usage example: {args.output_dir}/ml_ready/usage_example.py")
    
    print("\n💡 Next steps:")
    print("   1. Analyze the visualizations to understand data distribution")
    print("   2. Use ML-ready files for training classification models")
    print("   3. Experiment with different feature engineering approaches")
    print("   4. Consider ensemble methods for better performance")


def create_usage_example(output_dir: str):
    """Create a usage example script for the generated ML data"""
    
    usage_script = '''#!/usr/bin/env python3
"""
Usage Example: Maritime Radar Dataset for Machine Learning

This script demonstrates how to load and use the generated maritime radar dataset
for training machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib
import json

def load_dataset(data_type='detection'):
    """Load train/val/test datasets"""
    
    # Load datasets
    train_df = pd.read_hdf(f'{data_type}_train.h5', key='data')
    val_df = pd.read_hdf(f'{data_type}_val.h5', key='data')
    test_df = pd.read_hdf(f'{data_type}_test.h5', key='data')
    
    # Load feature info
    with open(f'{data_type}_feature_info.json', 'r') as f:
        feature_info = json.load(f)
    
    # Separate features and labels
    feature_cols = [col for col in train_df.columns if col != 'is_target']
    
    X_train = train_df[feature_cols]
    y_train = train_df['is_target']
    
    X_val = val_df[feature_cols]
    y_val = val_df['is_target']
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_target']
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_info

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest classifier"""
    
    print("Training Random Forest...")
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = rf.predict(X_val)
    val_score = rf.score(X_val, y_val)
    
    print(f"Validation Accuracy: {val_score:.3f}")
    print("\\nValidation Classification Report:")
    print(classification_report(y_val, val_pred))
    
    return rf

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Train Logistic Regression classifier"""
    
    print("Training Logistic Regression...")
    
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000
    )
    
    lr.fit(X_train, y_train)
    
    # Evaluate on validation set
    val_pred = lr.predict(X_val)
    val_score = lr.score(X_val, y_val)
    
    print(f"Validation Accuracy: {val_score:.3f}")
    print("\\nValidation Classification Report:")
    print(classification_report(y_val, val_pred))
    
    return lr

def main():
    """Main training pipeline"""
    
    print("="*60)
    print("MARITIME RADAR DATASET - ML EXAMPLE")
    print("="*60)
    
    # Load detection-level data
    print("Loading detection-level dataset...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_info = load_dataset('detection')
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {len(X_train.columns)}")
    print(f"Target ratio: {feature_info['class_balance']['target_ratio']:.3f}")
    
    # Train models
    print("\\n" + "="*40)
    print("TRAINING MODELS")
    print("="*40)
    
    # Random Forest
    rf_model = train_random_forest(X_train, y_train, X_val, y_val)
    
    print("\\n" + "-"*40)
    
    # Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # Final evaluation on test set
    print("\\n" + "="*40)
    print("TEST SET EVALUATION")
    print("="*40)
    
    # Random Forest test performance
    rf_test_pred = rf_model.predict(X_test)
    rf_test_score = rf_model.score(X_test, y_test)
    
    print(f"\\nRandom Forest Test Accuracy: {rf_test_score:.3f}")
    print("Random Forest Test Classification Report:")
    print(classification_report(y_test, rf_test_pred))
    
    # Logistic Regression test performance
    lr_test_pred = lr_model.predict(X_test)
    lr_test_score = lr_model.score(X_test, y_test)
    
    print(f"\\nLogistic Regression Test Accuracy: {lr_test_score:.3f}")
    print("Logistic Regression Test Classification Report:")
    print(classification_report(y_test, lr_test_pred))
    
    # Feature importance (Random Forest)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\\nTop 10 Most Important Features (Random Forest):")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save models
    joblib.dump(rf_model, 'random_forest_model.pkl')
    joblib.dump(lr_model, 'logistic_regression_model.pkl')
    
    print("\\n✅ Models saved:")
    print("   • random_forest_model.pkl")
    print("   • logistic_regression_model.pkl")

if __name__ == "__main__":
    main()
'''
    
    script_path = os.path.join(output_dir, "usage_example.py")
    with open(script_path, 'w') as f:
        f.write(usage_script)
    
    print(f"📝 Usage example created: {script_path}")


if __name__ == "__main__":
    main()