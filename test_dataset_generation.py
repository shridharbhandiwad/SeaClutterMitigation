#!/usr/bin/env python3
"""
Simple test for Maritime Radar Dataset Generation

This test demonstrates the core functionality without complex dependencies.
"""

import sys
import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Simulate basic dependencies if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("Warning: NumPy not available, using simplified generation")

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: Pandas not available, using basic CSV output")


@dataclass
class RadarDetection:
    """Single radar detection point"""
    track_id: str
    timestamp: str
    range_m: float
    azimuth_deg: float
    elevation_deg: float
    doppler_ms: float
    rcs_dbsm: float
    snr_db: float
    is_target: bool
    sea_state: int
    
    def to_dict(self):
        return asdict(self)


def simple_random(seed=None, size=1):
    """Simple random number generator if numpy unavailable"""
    if HAS_NUMPY:
        if seed is not None:
            np.random.seed(seed)
        return np.random.random(size) if size > 1 else np.random.random()
    
    # Simple linear congruential generator
    import time
    if seed is None:
        seed = int(time.time() * 1000) % 2**31
    
    values = []
    for _ in range(size if size > 1 else 1):
        seed = (seed * 1103515245 + 12345) % 2**31
        values.append(seed / 2**31)
    
    return values if size > 1 else values[0]


def simple_normal(mean=0, std=1, seed=None):
    """Simple normal distribution approximation"""
    if HAS_NUMPY:
        return np.random.normal(mean, std)
    
    # Box-Muller approximation using uniform random
    u1 = simple_random(seed)
    u2 = simple_random()
    
    # Approximate normal distribution
    import math
    z0 = math.sqrt(-2 * math.log(u1 + 1e-10)) * math.cos(2 * math.pi * u2)
    return mean + std * z0


def generate_simple_clutter_track(track_id, sea_state, n_detections=100):
    """Generate a simple sea clutter track"""
    detections = []
    
    # Fixed position with small variations
    base_range = simple_random() * 20000 + 5000  # 5-25 km
    base_azimuth = simple_random() * 360
    
    start_time = datetime.utcnow()
    
    for i in range(n_detections):
        # Small position variations due to wave motion
        wave_height = 0.5 + sea_state * 0.5  # Rough estimate
        range_var = simple_normal(0, wave_height)
        azimuth_var = simple_normal(0, 0.1)
        
        range_m = base_range + range_var
        azimuth_deg = (base_azimuth + azimuth_var) % 360
        elevation_deg = simple_normal(0, 1)
        
        # Doppler from wave motion
        doppler_ms = simple_normal(0, wave_height * 0.5)
        
        # RCS calculation (simplified)
        rcs_dbsm = -50 + 10 * math.log10(5 + sea_state * 2) + simple_normal(0, 3)
        
        # SNR calculation
        snr_db = rcs_dbsm - 30 + simple_normal(0, 5)
        
        # Timestamp
        dt = 0.001  # 1ms between detections
        timestamp = (start_time + timedelta(seconds=i * dt)).isoformat() + 'Z'
        
        detection = RadarDetection(
            track_id=track_id,
            timestamp=timestamp,
            range_m=range_m,
            azimuth_deg=azimuth_deg,
            elevation_deg=elevation_deg,
            doppler_ms=doppler_ms,
            rcs_dbsm=rcs_dbsm,
            snr_db=snr_db,
            is_target=False,
            sea_state=sea_state
        )
        
        detections.append(detection)
    
    return detections


def generate_simple_vessel_track(track_id, sea_state, duration_s=300):
    """Generate a simple vessel target track"""
    detections = []
    
    # Vessel parameters
    vessel_speed = 5 + simple_random() * 15  # 5-20 m/s
    base_heading = simple_random() * 2 * math.pi
    
    # Starting position
    x = simple_random() * 40000 - 20000  # ±20 km
    y = simple_random() * 29000 + 1000   # 1-30 km
    
    start_time = datetime.utcnow()
    dt = 0.01  # 10ms between detections
    n_detections = int(duration_s / dt)
    
    for i in range(n_detections):
        # Update position
        x += vessel_speed * math.cos(base_heading) * dt
        y += vessel_speed * math.sin(base_heading) * dt
        
        # Convert to polar
        range_m = math.sqrt(x**2 + y**2)
        azimuth_deg = math.degrees(math.atan2(x, y)) % 360
        
        # Skip if out of range
        if range_m > 50000 or range_m < 100:
            continue
        
        elevation_deg = simple_normal(0, 0.5)
        
        # Doppler from radial velocity
        radial_velocity = vessel_speed * (x * math.cos(base_heading) + y * math.sin(base_heading)) / (range_m + 1e-10)
        doppler_ms = radial_velocity + simple_normal(0, 0.5)
        
        # RCS for vessel
        base_rcs = simple_random() * 20 - 5  # -5 to 15 dBsm
        aspect_factor = 1 + 5 * abs(math.cos(math.radians(azimuth_deg)))
        rcs_dbsm = base_rcs + math.log10(aspect_factor) * 10 + simple_normal(0, 2)
        
        # SNR with range dependence
        range_loss_db = 40 * math.log10(range_m / 1000)
        snr_db = rcs_dbsm - range_loss_db + 50 + simple_normal(0, 3)
        
        # Timestamp
        timestamp = (start_time + timedelta(seconds=i * dt)).isoformat() + 'Z'
        
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
            sea_state=sea_state
        )
        
        detections.append(detection)
    
    return detections


def generate_test_dataset(output_dir="test_maritime_dataset", 
                         n_clutter_tracks=50, n_vessel_tracks=10):
    """Generate a small test dataset"""
    
    print(f"Generating test maritime radar dataset...")
    print(f"Output directory: {output_dir}")
    print(f"Clutter tracks: {n_clutter_tracks}")
    print(f"Vessel tracks: {n_vessel_tracks}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    all_detections = []
    
    # Generate clutter tracks
    print("\nGenerating clutter tracks...")
    for i in range(n_clutter_tracks):
        sea_state = int(simple_random() * 5) + 1  # 1-5
        track_id = f"CLUTTER_{i:04d}"
        n_detections = int(simple_random() * 100) + 50  # 50-150
        
        detections = generate_simple_clutter_track(track_id, sea_state, n_detections)
        all_detections.extend(detections)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_clutter_tracks} clutter tracks")
    
    # Generate vessel tracks
    print("\nGenerating vessel tracks...")
    for i in range(n_vessel_tracks):
        sea_state = int(simple_random() * 5) + 1  # 1-5
        track_id = f"VESSEL_{i:04d}"
        duration_s = simple_random() * 300 + 120  # 2-7 minutes
        
        detections = generate_simple_vessel_track(track_id, sea_state, duration_s)
        all_detections.extend(detections)
        
        if (i + 1) % 5 == 0:
            print(f"  Generated {i + 1}/{n_vessel_tracks} vessel tracks")
    
    print(f"\nTotal detections generated: {len(all_detections)}")
    
    # Save as CSV
    csv_file = os.path.join(output_dir, "maritime_radar_test.csv")
    with open(csv_file, 'w') as f:
        # Header
        headers = ["track_id", "timestamp", "range_m", "azimuth_deg", "elevation_deg",
                  "doppler_ms", "rcs_dbsm", "snr_db", "is_target", "sea_state"]
        f.write(",".join(headers) + "\n")
        
        # Data
        for detection in all_detections:
            row = [
                detection.track_id,
                detection.timestamp,
                f"{detection.range_m:.1f}",
                f"{detection.azimuth_deg:.2f}",
                f"{detection.elevation_deg:.2f}",
                f"{detection.doppler_ms:.2f}",
                f"{detection.rcs_dbsm:.1f}",
                f"{detection.snr_db:.1f}",
                str(detection.is_target).lower(),
                str(detection.sea_state)
            ]
            f.write(",".join(row) + "\n")
    
    # Save metadata
    metadata = {
        "dataset_info": {
            "name": "Test Maritime Radar Dataset",
            "version": "1.0",
            "generated_at": datetime.utcnow().isoformat() + 'Z',
            "description": "Simple test dataset for maritime radar detection"
        },
        "statistics": {
            "total_detections": len(all_detections),
            "clutter_tracks": n_clutter_tracks,
            "vessel_tracks": n_vessel_tracks,
            "target_detections": sum(1 for d in all_detections if d.is_target),
            "clutter_detections": sum(1 for d in all_detections if not d.is_target)
        }
    }
    
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Summary
    target_count = sum(1 for d in all_detections if d.is_target)
    clutter_count = len(all_detections) - target_count
    
    print(f"\n✅ Dataset generated successfully!")
    print(f"📁 Location: {output_dir}")
    print(f"📊 Statistics:")
    print(f"   • Total detections: {len(all_detections):,}")
    print(f"   • Target detections: {target_count:,}")
    print(f"   • Clutter detections: {clutter_count:,}")
    print(f"   • Target ratio: {target_count/len(all_detections):.3f}")
    print(f"📄 Files:")
    print(f"   • Data: {csv_file}")
    print(f"   • Metadata: {metadata_file}")
    
    # File size
    file_size = os.path.getsize(csv_file)
    print(f"   • Size: {file_size/1024/1024:.1f} MB")
    
    return csv_file


if __name__ == "__main__":
    import math
    
    print("=" * 60)
    print("MARITIME RADAR DATASET - SIMPLE TEST")
    print("=" * 60)
    
    # Test with small dataset
    dataset_file = generate_test_dataset(
        output_dir="test_maritime_dataset",
        n_clutter_tracks=100,
        n_vessel_tracks=20
    )
    
    print("\n💡 Next steps:")
    print("   1. Examine the generated CSV file")
    print("   2. Import into your preferred analysis tool")
    print("   3. Use for machine learning experiments")
    print("   4. Scale up with the full system when dependencies are available")
    
    print(f"\n🎯 Test completed successfully!")