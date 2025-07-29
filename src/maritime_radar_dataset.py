"""
Maritime Radar Dataset Generator

A comprehensive system for generating large-scale realistic maritime radar datasets
with labeled tracks for sea clutter and vessel targets under various conditions.

Features:
- Physically-based clutter models (Weibull, K-distribution)
- Realistic vessel target simulation with motion models
- Multi-track sequential detection generation
- Various sea states (calm, moderate, rough)
- Configurable dataset size and parameters
- Training/validation/test splits
"""

import numpy as np
import pandas as pd
import h5py
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import multiprocessing as mp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


@dataclass
class RadarDetection:
    """Single radar detection point"""
    track_id: str
    timestamp: str  # ISO-8601 UTC
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    doppler_ms: float
    rcs_dbsm: float
    snr_db: float
    is_target: bool  # True for target, False for clutter
    sea_state: int  # 0-9 scale
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RadarParameters:
    """Radar system parameters"""
    frequency_hz: float = 9.4e9  # X-band
    bandwidth_hz: float = 50e6
    pulse_width_s: float = 1e-6
    prf_hz: float = 1000
    antenna_gain_db: float = 35.0
    noise_figure_db: float = 3.0
    max_range_m: float = 50000
    azimuth_beamwidth_deg: float = 1.0
    elevation_beamwidth_deg: float = 20.0


@dataclass
class EnvironmentalParameters:
    """Environmental conditions"""
    sea_state: int = 3  # 0-9 scale
    wind_speed_ms: float = 10.0
    wave_height_m: float = 2.0
    temperature_c: float = 15.0
    humidity_percent: float = 75.0


class SeaClutterModel:
    """Physical sea clutter modeling using K-distribution and Weibull models"""
    
    def __init__(self, radar_params: RadarParameters, env_params: EnvironmentalParameters):
        self.radar_params = radar_params
        self.env_params = env_params
        
    def k_distribution_clutter(self, size: int, shape_param: float = 2.0) -> np.ndarray:
        """
        Generate sea clutter using K-distribution model
        
        Args:
            size: Number of samples
            shape_param: K-distribution shape parameter (depends on sea state)
            
        Returns:
            Complex clutter samples
        """
        # K-distribution parameters based on sea state
        k_params = {
            0: 5.0,   # Calm
            1: 4.0,   # Slight
            2: 3.0,   # Moderate
            3: 2.5,   # Rough
            4: 2.0,   # Very rough
            5: 1.5,   # High
            6: 1.0,   # Very high
            7: 0.8,   # Phenomenal
            8: 0.6,   # Extreme
            9: 0.4    # Hurricane
        }
        
        k = k_params.get(self.env_params.sea_state, 2.0)
        
        # Generate K-distributed amplitude
        # K-distribution can be modeled as Gamma-distributed power
        scale = 1.0
        amplitude = np.sqrt(np.random.gamma(k, scale/k, size))
        
        # Random phase
        phase = np.random.uniform(0, 2*np.pi, size)
        
        # Complex clutter
        clutter = amplitude * np.exp(1j * phase)
        
        # Add correlation for realistic sea clutter
        # Apply moving average for temporal correlation
        if len(clutter) > 10:
            kernel_size = min(5, len(clutter)//4)
            kernel = np.ones(kernel_size) / kernel_size
            amplitude_corr = np.convolve(np.abs(clutter), kernel, mode='same')
            clutter = amplitude_corr * np.exp(1j * phase)
        
        return clutter
    
    def weibull_clutter(self, size: int) -> np.ndarray:
        """
        Generate sea clutter using Weibull distribution
        
        Args:
            size: Number of samples
            
        Returns:
            Complex clutter samples
        """
        # Weibull parameters based on sea state
        weibull_params = {
            0: (2.5, 1.0),  # (shape, scale)
            1: (2.2, 1.1),
            2: (2.0, 1.2),
            3: (1.8, 1.3),
            4: (1.6, 1.4),
            5: (1.4, 1.5),
            6: (1.2, 1.6),
            7: (1.0, 1.7),
            8: (0.9, 1.8),
            9: (0.8, 1.9)
        }
        
        shape, scale = weibull_params.get(self.env_params.sea_state, (2.0, 1.0))
        
        # Generate Weibull-distributed amplitude
        amplitude = np.random.weibull(shape, size) * scale
        
        # Random phase
        phase = np.random.uniform(0, 2*np.pi, size)
        
        return amplitude * np.exp(1j * phase)
    
    def calculate_clutter_rcs(self, range_m: float, azimuth_deg: float) -> float:
        """Calculate sea clutter RCS using empirical models"""
        
        # Nathanson's clutter model
        sigma_0_db = -50 + 10 * np.log10(self.env_params.wind_speed_ms) + \
                     2 * self.env_params.sea_state
        
        # Range and angle dependencies
        range_factor = 10 * np.log10(range_m / 1000.0)  # Reference 1km
        angle_factor = -np.abs(azimuth_deg - 90) / 45.0  # Minimum at 90 degrees
        
        rcs_dbsm = sigma_0_db + range_factor + angle_factor
        
        # Add random variation
        rcs_dbsm += np.random.normal(0, 3)  # 3dB standard deviation
        
        return rcs_dbsm


class VesselTargetModel:
    """Vessel target modeling with realistic motion patterns"""
    
    def __init__(self, radar_params: RadarParameters):
        self.radar_params = radar_params
        
    def generate_vessel_motion(self, duration_s: float, dt_s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate realistic vessel motion pattern
        
        Args:
            duration_s: Track duration in seconds
            dt_s: Time step in seconds
            
        Returns:
            Tuple of (x_positions, y_positions, times)
        """
        n_samples = int(duration_s / dt_s)
        times = np.arange(n_samples) * dt_s
        
        # Vessel parameters
        vessel_types = {
            'small_boat': {'speed_ms': np.random.uniform(5, 15), 'maneuver_freq': 0.1},
            'fishing_vessel': {'speed_ms': np.random.uniform(3, 8), 'maneuver_freq': 0.05},
            'cargo_ship': {'speed_ms': np.random.uniform(8, 12), 'maneuver_freq': 0.02},
            'patrol_boat': {'speed_ms': np.random.uniform(10, 25), 'maneuver_freq': 0.15}
        }
        
        vessel_type = np.random.choice(list(vessel_types.keys()))
        params = vessel_types[vessel_type]
        
        # Base velocity
        base_speed = params['speed_ms']
        base_heading = np.random.uniform(0, 2*np.pi)
        
        # Initialize position
        x_pos = np.zeros(n_samples)
        y_pos = np.zeros(n_samples)
        
        # Starting position (random within radar coverage)
        x_pos[0] = np.random.uniform(-20000, 20000)
        y_pos[0] = np.random.uniform(1000, 30000)  # Keep away from radar
        
        # Generate motion with random maneuvers
        heading = base_heading
        speed = base_speed
        
        for i in range(1, n_samples):
            # Random maneuvers
            if np.random.random() < params['maneuver_freq'] * dt_s:
                # Course change
                heading += np.random.normal(0, np.pi/6)  # ±30 degrees
                # Speed change
                speed = base_speed * np.random.uniform(0.7, 1.3)
            
            # Update position
            x_pos[i] = x_pos[i-1] + speed * np.cos(heading) * dt_s
            y_pos[i] = y_pos[i-1] + speed * np.sin(heading) * dt_s
        
        return x_pos, y_pos, times
    
    def calculate_vessel_rcs(self, vessel_type: str = 'small_boat', 
                           azimuth_deg: float = 0) -> float:
        """Calculate vessel RCS based on type and aspect angle"""
        
        # Base RCS values for different vessel types (dBsm)
        base_rcs = {
            'small_boat': np.random.uniform(-5, 5),
            'fishing_vessel': np.random.uniform(5, 15),
            'cargo_ship': np.random.uniform(20, 35),
            'patrol_boat': np.random.uniform(0, 10)
        }
        
        rcs_base = base_rcs.get(vessel_type, 0)
        
        # Aspect angle dependency (simplified)
        aspect_factor = 1 + 5 * np.cos(np.radians(azimuth_deg))  # Max at bow/stern
        
        rcs_dbsm = rcs_base + 10 * np.log10(np.abs(aspect_factor))
        
        # Add random fluctuation (Swerling models)
        rcs_dbsm += np.random.normal(0, 2)  # 2dB standard deviation
        
        return rcs_dbsm


class MaritimeRadarDatasetGenerator:
    """Main class for generating large-scale maritime radar datasets"""
    
    def __init__(self, radar_params: RadarParameters = None, 
                 output_dir: str = "maritime_radar_dataset"):
        self.radar_params = radar_params or RadarParameters()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Dataset statistics
        self.stats = {
            'total_detections': 0,
            'target_detections': 0,
            'clutter_detections': 0,
            'tracks_generated': 0
        }
    
    def generate_clutter_track(self, track_id: str, env_params: EnvironmentalParameters,
                             n_detections: int = 100) -> List[RadarDetection]:
        """Generate a sea clutter track"""
        
        clutter_model = SeaClutterModel(self.radar_params, env_params)
        detections = []
        
        # Fixed position for clutter (varies slightly due to wave motion)
        base_range = np.random.uniform(1000, 30000)
        base_azimuth = np.random.uniform(0, 360)
        
        # Time parameters
        start_time = datetime.utcnow()
        dt = 1.0 / self.radar_params.prf_hz
        
        for i in range(n_detections):
            # Small position variations due to wave motion
            range_var = np.random.normal(0, env_params.wave_height_m)
            azimuth_var = np.random.normal(0, 0.1)  # Small angular spread
            
            range_m = base_range + range_var
            azimuth_deg = base_azimuth + azimuth_var
            elevation_deg = np.random.normal(0, 1)  # Sea level
            
            # Doppler from wave motion
            doppler_ms = np.random.normal(0, env_params.wave_height_m * 0.5)
            
            # RCS calculation
            rcs_dbsm = clutter_model.calculate_clutter_rcs(range_m, azimuth_deg)
            
            # SNR calculation (simplified)
            snr_db = rcs_dbsm - 30 + np.random.normal(0, 5)  # Base noise level
            
            # Timestamp
            timestamp = (start_time + timedelta(seconds=i * dt)).isoformat() + 'Z'
            
            detection = RadarDetection(
                track_id=track_id,
                timestamp=timestamp,
                range_m=range_m,
                azimuth_deg=azimuth_deg % 360,
                elevation_deg=elevation_deg,
                doppler_ms=doppler_ms,
                rcs_dbsm=rcs_dbsm,
                snr_db=snr_db,
                is_target=False,
                sea_state=env_params.sea_state
            )
            
            detections.append(detection)
        
        return detections
    
    def generate_vessel_track(self, track_id: str, env_params: EnvironmentalParameters,
                            duration_s: float = 300) -> List[RadarDetection]:
        """Generate a vessel target track"""
        
        vessel_model = VesselTargetModel(self.radar_params)
        detections = []
        
        # Generate vessel motion
        dt = 1.0 / self.radar_params.prf_hz * 10  # Sample every 10 pulses
        x_pos, y_pos, times = vessel_model.generate_vessel_motion(duration_s, dt)
        
        # Random vessel type
        vessel_types = ['small_boat', 'fishing_vessel', 'cargo_ship', 'patrol_boat']
        vessel_type = np.random.choice(vessel_types)
        
        start_time = datetime.utcnow()
        
        for i, (x, y, t) in enumerate(zip(x_pos, y_pos, times)):
            # Convert to polar coordinates
            range_m = np.sqrt(x**2 + y**2)
            azimuth_deg = np.degrees(np.arctan2(x, y)) % 360
            elevation_deg = np.random.normal(0, 0.5)  # Small elevation variation
            
            # Skip if out of radar range
            if range_m > self.radar_params.max_range_m or range_m < 100:
                continue
            
            # Calculate Doppler from velocity
            if i > 0:
                dx = x_pos[i] - x_pos[i-1]
                dy = y_pos[i] - y_pos[i-1]
                velocity_ms = np.sqrt(dx**2 + dy**2) / dt
                
                # Radial velocity component
                cos_angle = (x * dx + y * dy) / (range_m * np.sqrt(dx**2 + dy**2) + 1e-10)
                doppler_ms = velocity_ms * cos_angle
            else:
                doppler_ms = 0
            
            # RCS calculation
            aspect_angle = azimuth_deg
            rcs_dbsm = vessel_model.calculate_vessel_rcs(vessel_type, aspect_angle)
            
            # SNR calculation with range dependence
            range_loss_db = 40 * np.log10(range_m / 1000)  # 4th power law
            snr_db = rcs_dbsm - range_loss_db + 50 + np.random.normal(0, 3)
            
            # Timestamp
            timestamp = (start_time + timedelta(seconds=t)).isoformat() + 'Z'
            
            detection = RadarDetection(
                track_id=track_id,
                timestamp=timestamp,
                range_m=range_m,
                azimuth_deg=azimuth_deg,
                elevation_deg=elevation_deg,
                doppler_ms=doppler_ms,
                rcs_dbsm=rcs_dbsm,
                snr_db=snr_db,
                is_target=True,
                sea_state=env_params.sea_state
            )
            
            detections.append(detection)
        
        return detections
    
    def generate_dataset(self, n_clutter_tracks: int = 10000, 
                        n_vessel_tracks: int = 2000,
                        sea_states: List[int] = [1, 3, 5, 7],
                        min_detections_per_track: int = 50,
                        max_detections_per_track: int = 500,
                        use_multiprocessing: bool = True) -> str:
        """
        Generate large-scale maritime radar dataset
        
        Args:
            n_clutter_tracks: Number of clutter tracks
            n_vessel_tracks: Number of vessel tracks
            sea_states: List of sea states to include
            min_detections_per_track: Minimum detections per track
            max_detections_per_track: Maximum detections per track
            use_multiprocessing: Whether to use parallel processing
            
        Returns:
            Path to generated dataset file
        """
        
        print(f"Generating maritime radar dataset...")
        print(f"Clutter tracks: {n_clutter_tracks}")
        print(f"Vessel tracks: {n_vessel_tracks}")
        print(f"Sea states: {sea_states}")
        
        all_detections = []
        
        # Generate clutter tracks
        print("\nGenerating sea clutter tracks...")
        clutter_args = []
        for i in tqdm(range(n_clutter_tracks)):
            sea_state = np.random.choice(sea_states)
            env_params = EnvironmentalParameters(
                sea_state=sea_state,
                wind_speed_ms=np.random.uniform(5, 20),
                wave_height_m=np.random.uniform(0.5, 4.0)
            )
            
            n_detections = np.random.randint(min_detections_per_track, 
                                           max_detections_per_track)
            track_id = f"CLUTTER_{i:06d}"
            
            if use_multiprocessing:
                clutter_args.append((track_id, env_params, n_detections))
            else:
                detections = self.generate_clutter_track(track_id, env_params, n_detections)
                all_detections.extend(detections)
        
        if use_multiprocessing and clutter_args:
            with mp.Pool() as pool:
                clutter_results = pool.starmap(self.generate_clutter_track, clutter_args)
                for detections in clutter_results:
                    all_detections.extend(detections)
        
        # Generate vessel tracks
        print("\nGenerating vessel target tracks...")
        vessel_args = []
        for i in tqdm(range(n_vessel_tracks)):
            sea_state = np.random.choice(sea_states)
            env_params = EnvironmentalParameters(
                sea_state=sea_state,
                wind_speed_ms=np.random.uniform(5, 20),
                wave_height_m=np.random.uniform(0.5, 4.0)
            )
            
            duration_s = np.random.uniform(120, 600)  # 2-10 minutes
            track_id = f"VESSEL_{i:06d}"
            
            if use_multiprocessing:
                vessel_args.append((track_id, env_params, duration_s))
            else:
                detections = self.generate_vessel_track(track_id, env_params, duration_s)
                all_detections.extend(detections)
        
        if use_multiprocessing and vessel_args:
            with mp.Pool() as pool:
                vessel_results = pool.starmap(self.generate_vessel_track, vessel_args)
                for detections in vessel_results:
                    all_detections.extend(detections)
        
        # Convert to DataFrame and save
        print(f"\nCreated {len(all_detections)} total detections")
        
        # Update statistics
        self.stats['total_detections'] = len(all_detections)
        self.stats['target_detections'] = sum(1 for d in all_detections if d.is_target)
        self.stats['clutter_detections'] = sum(1 for d in all_detections if not d.is_target)
        self.stats['tracks_generated'] = len(set(d.track_id for d in all_detections))
        
        # Save dataset
        dataset_file = self._save_dataset(all_detections)
        
        # Save metadata
        self._save_metadata()
        
        print(f"\nDataset saved to: {dataset_file}")
        print(f"Dataset size: {os.path.getsize(dataset_file) / (1024**3):.2f} GB")
        
        return dataset_file
    
    def _save_dataset(self, detections: List[RadarDetection]) -> str:
        """Save dataset to multiple formats"""
        
        # Convert to DataFrame
        data_dicts = [d.to_dict() for d in detections]
        df = pd.DataFrame(data_dicts)
        
        # Sort by timestamp
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp_dt').drop('timestamp_dt', axis=1)
        
        # Save as HDF5 (efficient for large datasets)
        h5_file = os.path.join(self.output_dir, "maritime_radar_dataset.h5")
        df.to_hdf(h5_file, key='detections', mode='w', format='table')
        
        # Save as Parquet (good compression)
        parquet_file = os.path.join(self.output_dir, "maritime_radar_dataset.parquet")
        df.to_parquet(parquet_file, compression='snappy')
        
        # Save sample as CSV for inspection
        csv_file = os.path.join(self.output_dir, "maritime_radar_sample.csv")
        df.head(10000).to_csv(csv_file, index=False)
        
        return h5_file
    
    def _save_metadata(self):
        """Save dataset metadata and statistics"""
        
        metadata = {
            'dataset_info': {
                'name': 'Maritime Radar Dataset',
                'version': '1.0',
                'generated_at': datetime.utcnow().isoformat() + 'Z',
                'description': 'Large-scale synthetic maritime radar dataset with labeled tracks'
            },
            'radar_parameters': asdict(self.radar_params),
            'statistics': self.stats,
            'data_fields': {
                'track_id': 'Unique track identifier',
                'timestamp': 'ISO-8601 UTC timestamp',
                'range_m': 'Range in meters',
                'azimuth_deg': 'Azimuth in degrees',
                'elevation_deg': 'Elevation in degrees',
                'doppler_ms': 'Doppler velocity in m/s',
                'rcs_dbsm': 'Radar cross-section in dBsm',
                'snr_db': 'Signal-to-noise ratio in dB',
                'is_target': 'True for target, False for clutter',
                'sea_state': 'Sea state (0-9 scale)'
            }
        }
        
        metadata_file = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_file}")
    
    def create_train_val_test_split(self, test_ratio: float = 0.2, 
                                   val_ratio: float = 0.1) -> Dict[str, str]:
        """Create train/validation/test splits"""
        
        # Load full dataset
        dataset_file = os.path.join(self.output_dir, "maritime_radar_dataset.h5")
        df = pd.read_hdf(dataset_file, key='detections')
        
        # Get unique track IDs
        track_ids = df['track_id'].unique()
        n_tracks = len(track_ids)
        
        # Shuffle tracks
        np.random.shuffle(track_ids)
        
        # Split tracks
        n_test = int(n_tracks * test_ratio)
        n_val = int(n_tracks * val_ratio)
        n_train = n_tracks - n_test - n_val
        
        test_tracks = set(track_ids[:n_test])
        val_tracks = set(track_ids[n_test:n_test+n_val])
        train_tracks = set(track_ids[n_test+n_val:])
        
        # Create splits
        train_df = df[df['track_id'].isin(train_tracks)]
        val_df = df[df['track_id'].isin(val_tracks)]
        test_df = df[df['track_id'].isin(test_tracks)]
        
        # Save splits
        splits = {}
        for name, data in [('train', train_df), ('val', val_df), ('test', test_df)]:
            filename = f"maritime_radar_{name}.h5"
            filepath = os.path.join(self.output_dir, filename)
            data.to_hdf(filepath, key='detections', mode='w', format='table')
            splits[name] = filepath
            
            print(f"{name.upper()}: {len(data)} detections, {data['track_id'].nunique()} tracks")
        
        return splits


def generate_large_maritime_dataset(output_dir: str = "maritime_radar_dataset",
                                   target_size_gb: float = 2.0) -> str:
    """
    Generate a large maritime radar dataset targeting specific size
    
    Args:
        output_dir: Output directory
        target_size_gb: Target dataset size in GB
        
    Returns:
        Path to generated dataset
    """
    
    # Estimate tracks needed for target size
    # Rough estimate: 1000 detections per track, ~200 bytes per detection
    bytes_per_detection = 200
    detections_per_track = 200
    target_bytes = target_size_gb * 1024**3
    target_tracks = int(target_bytes / (detections_per_track * bytes_per_detection))
    
    # Split between clutter and targets (80% clutter, 20% targets)
    n_clutter_tracks = int(target_tracks * 0.8)
    n_vessel_tracks = int(target_tracks * 0.2)
    
    print(f"Targeting {target_size_gb} GB dataset")
    print(f"Estimated tracks needed: {target_tracks}")
    print(f"Clutter tracks: {n_clutter_tracks}")
    print(f"Vessel tracks: {n_vessel_tracks}")
    
    # Generate dataset
    generator = MaritimeRadarDatasetGenerator(output_dir=output_dir)
    dataset_file = generator.generate_dataset(
        n_clutter_tracks=n_clutter_tracks,
        n_vessel_tracks=n_vessel_tracks,
        sea_states=[1, 2, 3, 4, 5, 6, 7],
        min_detections_per_track=100,
        max_detections_per_track=500,
        use_multiprocessing=True
    )
    
    # Create train/val/test splits
    splits = generator.create_train_val_test_split()
    
    return dataset_file


if __name__ == "__main__":
    # Generate large dataset
    dataset_file = generate_large_maritime_dataset(
        output_dir="maritime_radar_dataset",
        target_size_gb=2.0
    )
    
    print(f"\nDataset generation complete!")
    print(f"Main file: {dataset_file}")