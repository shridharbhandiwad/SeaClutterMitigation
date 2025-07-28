"""
Maritime Radar Dataset Processor

Utilities for loading, processing, analyzing, and visualizing 
the generated maritime radar dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import h5py
import json
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')


class MaritimeRadarDatasetProcessor:
    """Processor for maritime radar dataset analysis and ML preparation"""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.df = None
        self.metadata = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load dataset and metadata"""
        try:
            # Load main dataset
            if self.dataset_path.endswith('.h5'):
                self.df = pd.read_hdf(self.dataset_path, key='detections')
            elif self.dataset_path.endswith('.parquet'):
                self.df = pd.read_parquet(self.dataset_path)
            elif self.dataset_path.endswith('.csv'):
                self.df = pd.read_csv(self.dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {self.dataset_path}")
            
            # Load metadata if available
            metadata_path = self.dataset_path.replace('.h5', '').replace('.parquet', '').replace('.csv', '')
            metadata_path = metadata_path.replace('maritime_radar_dataset', 'metadata.json')
            if '/maritime_radar_dataset' in metadata_path:
                metadata_path = metadata_path.replace('/maritime_radar_dataset', '/metadata.json')
            else:
                # Assume metadata is in the same directory
                import os
                dir_path = os.path.dirname(self.dataset_path)
                metadata_path = os.path.join(dir_path, 'metadata.json')
            
            try:
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            except FileNotFoundError:
                print("Metadata file not found, proceeding without metadata")
                self.metadata = {}
            
            print(f"Loaded dataset with {len(self.df)} detections")
            print(f"Unique tracks: {self.df['track_id'].nunique()}")
            print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            raise
    
    def get_dataset_summary(self) -> Dict:
        """Get comprehensive dataset summary"""
        
        summary = {
            'basic_stats': {
                'total_detections': len(self.df),
                'unique_tracks': self.df['track_id'].nunique(),
                'target_detections': sum(self.df['is_target']),
                'clutter_detections': sum(~self.df['is_target']),
                'target_ratio': sum(self.df['is_target']) / len(self.df),
                'unique_sea_states': sorted(self.df['sea_state'].unique().tolist()),
                'time_span_hours': (pd.to_datetime(self.df['timestamp'].max()) - 
                                  pd.to_datetime(self.df['timestamp'].min())).total_seconds() / 3600
            },
            'field_statistics': {},
            'track_statistics': {}
        }
        
        # Field statistics
        numeric_fields = ['range_m', 'azimuth_deg', 'elevation_deg', 'doppler_ms', 'rcs_dbsm', 'snr_db']
        for field in numeric_fields:
            if field in self.df.columns:
                summary['field_statistics'][field] = {
                    'mean': float(self.df[field].mean()),
                    'std': float(self.df[field].std()),
                    'min': float(self.df[field].min()),
                    'max': float(self.df[field].max()),
                    'median': float(self.df[field].median())
                }
        
        # Track statistics
        track_lengths = self.df.groupby('track_id').size()
        summary['track_statistics'] = {
            'mean_track_length': float(track_lengths.mean()),
            'median_track_length': float(track_lengths.median()),
            'min_track_length': int(track_lengths.min()),
            'max_track_length': int(track_lengths.max()),
            'std_track_length': float(track_lengths.std())
        }
        
        return summary
    
    def create_track_features(self) -> pd.DataFrame:
        """Create track-level features for analysis"""
        
        track_features = []
        
        for track_id in self.df['track_id'].unique():
            track_data = self.df[self.df['track_id'] == track_id].sort_values('timestamp')
            
            if len(track_data) < 2:
                continue
            
            # Basic track info
            features = {
                'track_id': track_id,
                'is_target': track_data['is_target'].iloc[0],
                'sea_state': track_data['sea_state'].iloc[0],
                'track_length': len(track_data),
                'duration_seconds': (pd.to_datetime(track_data['timestamp'].iloc[-1]) - 
                                   pd.to_datetime(track_data['timestamp'].iloc[0])).total_seconds()
            }
            
            # Statistical features
            for field in ['range_m', 'azimuth_deg', 'elevation_deg', 'doppler_ms', 'rcs_dbsm', 'snr_db']:
                if field in track_data.columns:
                    values = track_data[field].values
                    features.update({
                        f'{field}_mean': np.mean(values),
                        f'{field}_std': np.std(values),
                        f'{field}_min': np.min(values),
                        f'{field}_max': np.max(values),
                        f'{field}_range': np.max(values) - np.min(values)
                    })
            
            # Motion features
            if len(track_data) > 2:
                ranges = track_data['range_m'].values
                azimuths = track_data['azimuth_deg'].values
                
                # Convert to Cartesian for velocity calculation
                x = ranges * np.sin(np.radians(azimuths))
                y = ranges * np.cos(np.radians(azimuths))
                
                # Calculate velocities
                dt = np.diff(pd.to_datetime(track_data['timestamp']).astype(np.int64)) / 1e9  # seconds
                dx = np.diff(x)
                dy = np.diff(y)
                velocities = np.sqrt(dx**2 + dy**2) / (dt + 1e-10)
                
                features.update({
                    'mean_velocity_ms': np.mean(velocities),
                    'std_velocity_ms': np.std(velocities),
                    'max_velocity_ms': np.max(velocities),
                    'total_distance_m': np.sum(np.sqrt(dx**2 + dy**2))
                })
                
                # Course changes
                course_changes = np.abs(np.diff(azimuths))
                course_changes = np.minimum(course_changes, 360 - course_changes)  # Handle wrap-around
                features.update({
                    'mean_course_change_deg': np.mean(course_changes),
                    'std_course_change_deg': np.std(course_changes),
                    'max_course_change_deg': np.max(course_changes)
                })
            
            track_features.append(features)
        
        return pd.DataFrame(track_features)
    
    def visualize_dataset_overview(self, save_path: str = None) -> go.Figure:
        """Create comprehensive dataset overview visualization"""
        
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=[
                'Target vs Clutter Distribution',
                'Sea State Distribution',
                'Range Distribution',
                'Azimuth Distribution',
                'Doppler Distribution',
                'RCS Distribution',
                'SNR Distribution',
                'Track Length Distribution',
                'Detection Timeline'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}, {"type": "histogram"}],
                   [{"type": "scatter"}, {"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # 1. Target vs Clutter pie chart
        target_counts = self.df['is_target'].value_counts()
        fig.add_trace(
            go.Pie(labels=['Clutter', 'Target'], values=[target_counts[False], target_counts[True]],
                   name="Target Distribution"),
            row=1, col=1
        )
        
        # 2. Sea state distribution
        sea_state_counts = self.df['sea_state'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=sea_state_counts.index, y=sea_state_counts.values, name="Sea State"),
            row=1, col=2
        )
        
        # 3. Range distribution
        fig.add_trace(
            go.Histogram(x=self.df['range_m']/1000, nbinsx=50, name="Range (km)"),
            row=1, col=3
        )
        
        # 4. Azimuth scatter (polar-like)
        sample_df = self.df.sample(n=min(5000, len(self.df)))
        fig.add_trace(
            go.Scattergl(
                x=sample_df['azimuth_deg'], 
                y=sample_df['range_m']/1000,
                mode='markers',
                marker=dict(color=sample_df['is_target'], colorscale='Viridis', size=3),
                name="Azimuth vs Range"
            ),
            row=2, col=1
        )
        
        # 5. Doppler distribution
        fig.add_trace(
            go.Histogram(x=self.df['doppler_ms'], nbinsx=50, name="Doppler (m/s)"),
            row=2, col=2
        )
        
        # 6. RCS distribution
        fig.add_trace(
            go.Histogram(x=self.df['rcs_dbsm'], nbinsx=50, name="RCS (dBsm)"),
            row=2, col=3
        )
        
        # 7. SNR distribution
        fig.add_trace(
            go.Histogram(x=self.df['snr_db'], nbinsx=50, name="SNR (dB)"),
            row=3, col=1
        )
        
        # 8. Track length distribution
        track_lengths = self.df.groupby('track_id').size()
        fig.add_trace(
            go.Histogram(x=track_lengths, nbinsx=50, name="Track Length"),
            row=3, col=2
        )
        
        # 9. Detection timeline
        self.df['timestamp_dt'] = pd.to_datetime(self.df['timestamp'])
        hourly_counts = self.df.set_index('timestamp_dt').resample('H').size()
        fig.add_trace(
            go.Scatter(x=hourly_counts.index, y=hourly_counts.values, 
                      mode='lines', name="Detections per Hour"),
            row=3, col=3
        )
        
        fig.update_layout(
            height=1200, 
            title_text="Maritime Radar Dataset Overview",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def visualize_track_analysis(self, n_sample_tracks: int = 20, save_path: str = None) -> go.Figure:
        """Visualize sample tracks and their characteristics"""
        
        # Sample tracks for visualization
        target_tracks = self.df[self.df['is_target']]['track_id'].unique()
        clutter_tracks = self.df[~self.df['is_target']]['track_id'].unique()
        
        sample_target_tracks = np.random.choice(target_tracks, size=min(n_sample_tracks//2, len(target_tracks)), replace=False)
        sample_clutter_tracks = np.random.choice(clutter_tracks, size=min(n_sample_tracks//2, len(clutter_tracks)), replace=False)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Sample Track Trajectories',
                'Track Velocity Profiles',
                'Track RCS Profiles',
                'Track SNR Profiles'
            ],
            specs=[[{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        colors = ['red' if 'VESSEL' in track_id else 'blue' 
                 for track_id in list(sample_target_tracks) + list(sample_clutter_tracks)]
        
        # 1. Track trajectories
        for i, track_id in enumerate(list(sample_target_tracks) + list(sample_clutter_tracks)):
            track_data = self.df[self.df['track_id'] == track_id].sort_values('timestamp')
            
            # Convert to Cartesian coordinates
            x = track_data['range_m'] * np.sin(np.radians(track_data['azimuth_deg']))
            y = track_data['range_m'] * np.cos(np.radians(track_data['azimuth_deg']))
            
            fig.add_trace(
                go.Scatter(
                    x=x/1000, y=y/1000, 
                    mode='lines+markers',
                    name=f"{'Target' if 'VESSEL' in track_id else 'Clutter'} {i}",
                    line=dict(color=colors[i]),
                    showlegend=(i < 4)  # Only show legend for first few
                ),
                row=1, col=1
            )
        
        # 2. Velocity profiles
        for i, track_id in enumerate(list(sample_target_tracks)[:5]):  # Just show targets
            track_data = self.df[self.df['track_id'] == track_id].sort_values('timestamp')
            if len(track_data) > 2:
                times = pd.to_datetime(track_data['timestamp'])
                ranges = track_data['range_m'].values
                azimuths = track_data['azimuth_deg'].values
                
                x = ranges * np.sin(np.radians(azimuths))
                y = ranges * np.cos(np.radians(azimuths))
                
                dt = np.diff(times.astype(np.int64)) / 1e9
                dx = np.diff(x)
                dy = np.diff(y)
                velocities = np.sqrt(dx**2 + dy**2) / (dt + 1e-10)
                
                fig.add_trace(
                    go.Scatter(
                        x=times[1:], y=velocities,
                        mode='lines',
                        name=f"Vessel {i}",
                        showlegend=False
                    ),
                    row=1, col=2
                )
        
        # 3. RCS profiles
        for i, track_id in enumerate(list(sample_target_tracks)[:5]):
            track_data = self.df[self.df['track_id'] == track_id].sort_values('timestamp')
            times = pd.to_datetime(track_data['timestamp'])
            
            fig.add_trace(
                go.Scatter(
                    x=times, y=track_data['rcs_dbsm'],
                    mode='lines',
                    name=f"Vessel {i}",
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. SNR profiles
        for i, track_id in enumerate(list(sample_target_tracks)[:5]):
            track_data = self.df[self.df['track_id'] == track_id].sort_values('timestamp')
            times = pd.to_datetime(track_data['timestamp'])
            
            fig.add_trace(
                go.Scatter(
                    x=times, y=track_data['snr_db'],
                    mode='lines',
                    name=f"Vessel {i}",
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="Track Analysis Visualization"
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="X (km)", row=1, col=1)
        fig.update_yaxes(title_text="Y (km)", row=1, col=1)
        fig.update_yaxes(title_text="Velocity (m/s)", row=1, col=2)
        fig.update_yaxes(title_text="RCS (dBsm)", row=2, col=1)
        fig.update_yaxes(title_text="SNR (dB)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def prepare_ml_features(self, feature_type: str = 'detection') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning
        
        Args:
            feature_type: 'detection' for per-detection features, 'track' for track-level features
            
        Returns:
            Tuple of (features_df, labels)
        """
        
        if feature_type == 'detection':
            # Per-detection features
            feature_columns = ['range_m', 'azimuth_deg', 'elevation_deg', 'doppler_ms', 
                             'rcs_dbsm', 'snr_db', 'sea_state']
            
            features = self.df[feature_columns].copy()
            
            # Add derived features
            features['range_km'] = features['range_m'] / 1000
            features['azimuth_rad'] = np.radians(features['azimuth_deg'])
            features['elevation_rad'] = np.radians(features['elevation_deg'])
            
            # Cartesian coordinates
            features['x_coord'] = features['range_m'] * np.sin(features['azimuth_rad'])
            features['y_coord'] = features['range_m'] * np.cos(features['azimuth_rad'])
            
            labels = self.df['is_target'].astype(int)
            
        elif feature_type == 'track':
            # Track-level features
            track_features_df = self.create_track_features()
            
            # Select relevant columns (exclude track_id)
            feature_columns = [col for col in track_features_df.columns 
                             if col not in ['track_id', 'is_target']]
            
            features = track_features_df[feature_columns].copy()
            labels = track_features_df['is_target'].astype(int)
            
        else:
            raise ValueError("feature_type must be 'detection' or 'track'")
        
        # Handle missing values
        features = features.fillna(features.median())
        
        return features, labels
    
    def export_for_training(self, output_dir: str = "ml_ready_data", 
                           test_size: float = 0.2, val_size: float = 0.1,
                           feature_type: str = 'detection',
                           normalize: bool = True) -> Dict[str, str]:
        """
        Export dataset ready for ML training
        
        Args:
            output_dir: Output directory
            test_size: Test set ratio
            val_size: Validation set ratio
            feature_type: 'detection' or 'track'
            normalize: Whether to normalize features
            
        Returns:
            Dictionary with file paths
        """
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare features
        features, labels = self.prepare_ml_features(feature_type)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=42
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), stratify=y_temp, random_state=42
        )
        
        # Normalize if requested
        if normalize:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler
            import joblib
            scaler_path = os.path.join(output_dir, f'{feature_type}_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            
            # Use scaled data
            X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
            X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
            X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Save datasets
        file_paths = {}
        
        for split_name, X_data, y_data in [('train', X_train, y_train), 
                                          ('val', X_val, y_val), 
                                          ('test', X_test, y_test)]:
            
            # Combine features and labels
            combined_df = X_data.copy()
            combined_df['is_target'] = y_data.values
            
            # Save as HDF5
            h5_path = os.path.join(output_dir, f'{feature_type}_{split_name}.h5')
            combined_df.to_hdf(h5_path, key='data', mode='w')
            file_paths[split_name] = h5_path
            
            print(f"{split_name.upper()}: {len(combined_df)} samples, "
                  f"{sum(y_data)} targets ({sum(y_data)/len(y_data)*100:.1f}%)")
        
        # Save feature names
        feature_info = {
            'feature_names': list(features.columns),
            'feature_type': feature_type,
            'normalized': normalize,
            'n_features': len(features.columns),
            'n_samples': len(features),
            'class_balance': {
                'targets': int(sum(labels)),
                'clutter': int(len(labels) - sum(labels)),
                'target_ratio': float(sum(labels) / len(labels))
            }
        }
        
        feature_info_path = os.path.join(output_dir, f'{feature_type}_feature_info.json')
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        file_paths['feature_info'] = feature_info_path
        if normalize:
            file_paths['scaler'] = scaler_path
        
        return file_paths


def load_and_analyze_dataset(dataset_path: str, create_visualizations: bool = True) -> MaritimeRadarDatasetProcessor:
    """Convenience function to load and analyze dataset"""
    
    processor = MaritimeRadarDatasetProcessor(dataset_path)
    
    # Print summary
    summary = processor.get_dataset_summary()
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    print(f"Total detections: {summary['basic_stats']['total_detections']:,}")
    print(f"Unique tracks: {summary['basic_stats']['unique_tracks']:,}")
    print(f"Target detections: {summary['basic_stats']['target_detections']:,}")
    print(f"Clutter detections: {summary['basic_stats']['clutter_detections']:,}")
    print(f"Target ratio: {summary['basic_stats']['target_ratio']:.3f}")
    print(f"Sea states: {summary['basic_stats']['unique_sea_states']}")
    print(f"Time span: {summary['basic_stats']['time_span_hours']:.1f} hours")
    
    print(f"\nMean track length: {summary['track_statistics']['mean_track_length']:.1f}")
    print(f"Range statistics (km): {summary['field_statistics']['range_m']['min']/1000:.1f} - {summary['field_statistics']['range_m']['max']/1000:.1f}")
    print(f"RCS statistics (dBsm): {summary['field_statistics']['rcs_dbsm']['min']:.1f} - {summary['field_statistics']['rcs_dbsm']['max']:.1f}")
    
    if create_visualizations:
        print("\nCreating visualizations...")
        overview_fig = processor.visualize_dataset_overview()
        track_fig = processor.visualize_track_analysis()
        
        overview_fig.show()
        track_fig.show()
    
    return processor


if __name__ == "__main__":
    # Example usage
    dataset_path = "maritime_radar_dataset/maritime_radar_dataset.h5"
    processor = load_and_analyze_dataset(dataset_path)
    
    # Export for ML training
    ml_files = processor.export_for_training(feature_type='detection')
    print(f"\nML-ready files saved: {list(ml_files.keys())}")