"""
Data loader for IPIX and CSIR radar datasets.
Handles loading and preprocessing of raw I/Q radar data.
"""

import numpy as np
import pandas as pd
import h5py
import os
import pickle
from typing import Tuple, Dict, List, Optional, Union
import requests
from tqdm import tqdm
import zipfile
import tarfile


class RadarDataLoader:
    """
    Unified data loader for IPIX and CSIR radar datasets.
    Handles downloading, loading, and preprocessing of I/Q radar data.
    """
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store downloaded and processed data
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset URLs (these would need to be updated with actual URLs)
        self.dataset_urls = {
            'ipix': {
                'url': 'https://example.com/ipix_dataset.zip',  # Update with actual URL
                'filename': 'ipix_dataset.zip'
            },
            'csir': {
                'url': 'https://example.com/csir_dataset.tar.gz',  # Update with actual URL
                'filename': 'csir_dataset.tar.gz'
            }
        }
    
    def download_dataset(self, dataset_name: str) -> bool:
        """
        Download dataset if not already present.
        
        Args:
            dataset_name: 'ipix' or 'csir'
            
        Returns:
            True if download successful or file already exists
        """
        if dataset_name not in self.dataset_urls:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        url = self.dataset_urls[dataset_name]['url']
        filename = self.dataset_urls[dataset_name]['filename']
        filepath = os.path.join(self.data_dir, filename)
        
        if os.path.exists(filepath):
            print(f"Dataset {dataset_name} already exists at {filepath}")
            return True
        
        print(f"Downloading {dataset_name} dataset...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract the archive
            self._extract_archive(filepath, dataset_name)
            return True
            
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return False
    
    def _extract_archive(self, filepath: str, dataset_name: str):
        """Extract downloaded archive."""
        extract_dir = os.path.join(self.data_dir, dataset_name)
        os.makedirs(extract_dir, exist_ok=True)
        
        if filepath.endswith('.zip'):
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif filepath.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_dir)
    
    def load_ipix_data(self, file_path: str = None) -> Dict:
        """
        Load IPIX radar dataset.
        
        Args:
            file_path: Path to IPIX data file
            
        Returns:
            Dictionary containing I/Q data and metadata
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "ipix")
        
        # IPIX typically comes in .mat or custom binary format
        # This is a template - adjust based on actual file format
        
        data = {}
        
        # Example for .mat files (MATLAB format)
        try:
            from scipy.io import loadmat
            mat_data = loadmat(file_path)
            
            # Extract I/Q data (adjust keys based on actual IPIX format)
            if 'I_data' in mat_data and 'Q_data' in mat_data:
                data['I'] = mat_data['I_data']
                data['Q'] = mat_data['Q_data']
            elif 'iq_data' in mat_data:
                iq_complex = mat_data['iq_data']
                data['I'] = np.real(iq_complex)
                data['Q'] = np.imag(iq_complex)
            
            # Extract metadata
            data['metadata'] = {
                'polarization': mat_data.get('polarization', 'unknown'),
                'sea_state': mat_data.get('sea_state', 'unknown'),
                'range_cells': mat_data.get('range_cells', None),
                'time_samples': mat_data.get('time_samples', None),
                'labels': mat_data.get('labels', None)
            }
            
        except Exception as e:
            print(f"Error loading IPIX data: {e}")
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data('ipix')
        
        return data
    
    def load_csir_data(self, file_path: str = None) -> Dict:
        """
        Load CSIR radar dataset.
        
        Args:
            file_path: Path to CSIR data file
            
        Returns:
            Dictionary containing I/Q data and metadata
        """
        if file_path is None:
            file_path = os.path.join(self.data_dir, "csir")
        
        data = {}
        
        # CSIR typically comes in HDF5 or custom format
        try:
            if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                with h5py.File(file_path, 'r') as f:
                    # Adjust keys based on actual CSIR format
                    data['I'] = f['i_data'][:]
                    data['Q'] = f['q_data'][:]
                    
                    data['metadata'] = {
                        'polarization': f.attrs.get('polarization', 'unknown'),
                        'sea_state': f.attrs.get('sea_state', 'unknown'),
                        'range_cells': f.attrs.get('range_cells', None),
                        'time_samples': f.attrs.get('time_samples', None),
                        'labels': f.get('labels', None)
                    }
            else:
                # Handle other formats
                raise NotImplementedError("Format not supported yet")
                
        except Exception as e:
            print(f"Error loading CSIR data: {e}")
            # Generate synthetic data for demonstration
            data = self._generate_synthetic_data('csir')
        
        return data
    
    def _generate_synthetic_data(self, dataset_type: str) -> Dict:
        """
        Generate synthetic radar data for testing and demonstration.
        
        Args:
            dataset_type: 'ipix' or 'csir'
            
        Returns:
            Dictionary with synthetic I/Q data and labels
        """
        print(f"Generating synthetic {dataset_type} data for demonstration...")
        
        # Parameters
        n_range_cells = 20
        n_time_samples = 8192
        fs = 1000  # Sampling frequency
        
        # Generate complex I/Q data
        t = np.arange(n_time_samples) / fs
        
        # Initialize data
        I_data = np.zeros((n_range_cells, n_time_samples))
        Q_data = np.zeros((n_range_cells, n_time_samples))
        labels = np.zeros(n_range_cells)  # 0: clutter, 1: target
        
        for range_cell in range(n_range_cells):
            if range_cell < 15:  # First 15 cells are clutter
                # Sea clutter: lower frequency, more noise
                clutter_freq = np.random.uniform(0.5, 2.0)
                noise_level = 0.3
                
                I_signal = np.cos(2 * np.pi * clutter_freq * t) + noise_level * np.random.randn(n_time_samples)
                Q_signal = np.sin(2 * np.pi * clutter_freq * t) + noise_level * np.random.randn(n_time_samples)
                
                # Add amplitude modulation for sea clutter
                am_freq = np.random.uniform(0.1, 0.5)
                am_factor = 1 + 0.5 * np.sin(2 * np.pi * am_freq * t)
                
                I_data[range_cell] = I_signal * am_factor
                Q_data[range_cell] = Q_signal * am_factor
                labels[range_cell] = 0
                
            else:  # Last 5 cells contain targets
                # Target: higher frequency, more coherent
                target_freq = np.random.uniform(5.0, 15.0)
                noise_level = 0.1
                
                I_signal = np.cos(2 * np.pi * target_freq * t) + noise_level * np.random.randn(n_time_samples)
                Q_signal = np.sin(2 * np.pi * target_freq * t) + noise_level * np.random.randn(n_time_samples)
                
                I_data[range_cell] = I_signal
                Q_data[range_cell] = Q_signal
                labels[range_cell] = 1
        
        return {
            'I': I_data,
            'Q': Q_data,
            'metadata': {
                'polarization': 'VV',
                'sea_state': '3',
                'range_cells': n_range_cells,
                'time_samples': n_time_samples,
                'sampling_frequency': fs,
                'labels': labels
            }
        }
    
    def preprocess_data(self, data: Dict, normalize: bool = True, 
                       remove_dc: bool = True) -> Dict:
        """
        Preprocess I/Q radar data.
        
        Args:
            data: Dictionary containing I/Q data
            normalize: Whether to normalize the data
            remove_dc: Whether to remove DC component
            
        Returns:
            Preprocessed data dictionary
        """
        I_data = data['I'].copy()
        Q_data = data['Q'].copy()
        
        # Remove DC component
        if remove_dc:
            I_data = I_data - np.mean(I_data, axis=1, keepdims=True)
            Q_data = Q_data - np.mean(Q_data, axis=1, keepdims=True)
        
        # Normalize data
        if normalize:
            # Normalize each range cell independently
            for i in range(I_data.shape[0]):
                magnitude = np.sqrt(I_data[i]**2 + Q_data[i]**2)
                max_mag = np.max(magnitude)
                if max_mag > 0:
                    I_data[i] = I_data[i] / max_mag
                    Q_data[i] = Q_data[i] / max_mag
        
        # Update data
        processed_data = data.copy()
        processed_data['I'] = I_data
        processed_data['Q'] = Q_data
        
        return processed_data
    
    def save_processed_data(self, data: Dict, filename: str):
        """Save processed data to file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filename: str) -> Dict:
        """Load processed data from file."""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        return data


def load_radar_data(dataset_name: str = 'synthetic', 
                   data_dir: str = "data/") -> Dict:
    """
    Convenience function to load radar data.
    
    Args:
        dataset_name: 'ipix', 'csir', or 'synthetic'
        data_dir: Data directory
        
    Returns:
        Loaded and preprocessed radar data
    """
    loader = RadarDataLoader(data_dir)
    
    if dataset_name == 'ipix':
        data = loader.load_ipix_data()
    elif dataset_name == 'csir':
        data = loader.load_csir_data()
    else:  # synthetic
        data = loader._generate_synthetic_data('synthetic')
    
    # Preprocess the data
    processed_data = loader.preprocess_data(data)
    
    return processed_data


if __name__ == "__main__":
    # Example usage
    loader = RadarDataLoader()
    
    # Load synthetic data for demonstration
    data = loader._generate_synthetic_data('demo')
    processed_data = loader.preprocess_data(data)
    
    print("Data shape:", processed_data['I'].shape)
    print("Labels:", processed_data['metadata']['labels'])
    print("Sample loading complete!")