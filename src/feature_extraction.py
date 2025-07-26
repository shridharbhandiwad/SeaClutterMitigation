"""
Feature extraction for radar clutter classification.
Includes spectral, temporal, and statistical features for target/clutter discrimination.
"""

import numpy as np
from scipy import stats, signal
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
import cv2
from typing import Dict, List, Tuple, Optional, Union
import warnings


class RadarFeatureExtractor:
    """
    Feature extraction class for radar I/Q data.
    Extracts spectral, temporal, and statistical features for classification.
    """
    
    def __init__(self, sampling_frequency: float = 1000.0):
        """
        Initialize the feature extractor.
        
        Args:
            sampling_frequency: Sampling frequency in Hz
        """
        self.fs = sampling_frequency
        self.scaler = StandardScaler()
    
    def extract_spectral_features(self, spectrum: np.ndarray, 
                                frequencies: np.ndarray) -> Dict[str, float]:
        """
        Extract features from Doppler spectrum.
        
        Args:
            spectrum: Magnitude spectrum
            frequencies: Frequency array
            
        Returns:
            Dictionary of spectral features
        """
        # Normalize spectrum
        spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
        
        features = {}
        
        # Basic statistical features
        features['mean_magnitude'] = np.mean(spectrum)
        features['var_magnitude'] = np.var(spectrum)
        features['std_magnitude'] = np.std(spectrum)
        features['max_magnitude'] = np.max(spectrum)
        features['min_magnitude'] = np.min(spectrum)
        
        # Spectral centroid (weighted mean frequency)
        features['spectral_centroid'] = np.sum(frequencies * spectrum_norm)
        
        # Spectral spread (weighted standard deviation)
        features['spectral_spread'] = np.sqrt(
            np.sum(((frequencies - features['spectral_centroid']) ** 2) * spectrum_norm)
        )
        
        # Spectral skewness
        if features['spectral_spread'] > 0:
            features['spectral_skewness'] = np.sum(
                ((frequencies - features['spectral_centroid']) ** 3) * spectrum_norm
            ) / (features['spectral_spread'] ** 3)
        else:
            features['spectral_skewness'] = 0
        
        # Spectral kurtosis
        if features['spectral_spread'] > 0:
            features['spectral_kurtosis'] = np.sum(
                ((frequencies - features['spectral_centroid']) ** 4) * spectrum_norm
            ) / (features['spectral_spread'] ** 4)
        else:
            features['spectral_kurtosis'] = 0
        
        # Spectral entropy
        features['spectral_entropy'] = entropy(spectrum_norm + 1e-10)
        
        # Spectral flatness (geometric mean / arithmetic mean)
        geometric_mean = stats.gmean(spectrum + 1e-10)
        arithmetic_mean = np.mean(spectrum)
        features['spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
        
        # Peak frequency (frequency with maximum magnitude)
        peak_idx = np.argmax(spectrum)
        features['peak_frequency'] = frequencies[peak_idx]
        
        # Bandwidth (frequency range containing 90% of energy)
        cumulative_energy = np.cumsum(spectrum_norm)
        idx_5 = np.argmax(cumulative_energy >= 0.05)
        idx_95 = np.argmax(cumulative_energy >= 0.95)
        features['bandwidth_90'] = frequencies[idx_95] - frequencies[idx_5]
        
        # Bandwidth (frequency range containing 50% of energy)
        idx_25 = np.argmax(cumulative_energy >= 0.25)
        idx_75 = np.argmax(cumulative_energy >= 0.75)
        features['bandwidth_50'] = frequencies[idx_75] - frequencies[idx_25]
        
        # Number of spectral peaks
        peaks, _ = signal.find_peaks(spectrum, height=np.max(spectrum) * 0.1)
        features['num_peaks'] = len(peaks)
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        idx_85 = np.argmax(cumulative_energy >= 0.85)
        features['spectral_rolloff'] = frequencies[idx_85]
        
        # Zero crossing rate in spectrum
        features['spectral_zcr'] = np.sum(np.diff(np.sign(spectrum - np.mean(spectrum))) != 0)
        
        return features
    
    def extract_temporal_features(self, iq_data: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal features from I/Q time series.
        
        Args:
            iq_data: Complex I/Q data
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        
        # Magnitude and phase
        magnitude = np.abs(iq_data)
        phase = np.angle(iq_data)
        
        # Basic statistical features on magnitude
        features['mag_mean'] = np.mean(magnitude)
        features['mag_var'] = np.var(magnitude)
        features['mag_std'] = np.std(magnitude)
        features['mag_skewness'] = stats.skew(magnitude)
        features['mag_kurtosis'] = stats.kurtosis(magnitude)
        features['mag_max'] = np.max(magnitude)
        features['mag_min'] = np.min(magnitude)
        
        # Phase features
        phase_unwrapped = np.unwrap(phase)
        features['phase_var'] = np.var(phase_unwrapped)
        features['phase_std'] = np.std(phase_unwrapped)
        
        # Instantaneous frequency
        inst_freq = np.diff(phase_unwrapped) * self.fs / (2 * np.pi)
        features['inst_freq_mean'] = np.mean(inst_freq)
        features['inst_freq_var'] = np.var(inst_freq)
        features['inst_freq_std'] = np.std(inst_freq)
        
        # Zero crossing rate
        features['zcr_magnitude'] = np.sum(np.diff(np.sign(magnitude - np.mean(magnitude))) != 0)
        features['zcr_real'] = np.sum(np.diff(np.sign(np.real(iq_data))) != 0)
        features['zcr_imag'] = np.sum(np.diff(np.sign(np.imag(iq_data))) != 0)
        
        # Energy and power
        features['energy'] = np.sum(magnitude ** 2)
        features['power'] = features['energy'] / len(magnitude)
        
        # Peak-to-average power ratio
        features['papr'] = np.max(magnitude ** 2) / (features['power'] + 1e-10)
        
        # Crest factor (peak to RMS ratio)
        rms = np.sqrt(np.mean(magnitude ** 2))
        features['crest_factor'] = np.max(magnitude) / (rms + 1e-10)
        
        # Autocorrelation features
        autocorr = np.correlate(magnitude, magnitude, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first zero crossing in autocorrelation
        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        if len(zero_crossings) > 0:
            features['autocorr_first_zero'] = zero_crossings[0]
        else:
            features['autocorr_first_zero'] = len(autocorr)
        
        # Autocorrelation at lag 1
        if len(autocorr) > 1:
            features['autocorr_lag1'] = autocorr[1]
        else:
            features['autocorr_lag1'] = 0
        
        return features
    
    def extract_lbp_features(self, tds_image: np.ndarray, 
                           radius: int = 1, n_points: int = 8) -> Dict[str, float]:
        """
        Extract Local Binary Pattern (LBP) features from TDS image.
        
        Args:
            tds_image: Time-Doppler Spectrum image
            radius: LBP radius
            n_points: Number of LBP points
            
        Returns:
            Dictionary of LBP features
        """
        features = {}
        
        # Normalize image to 0-255
        if np.max(tds_image) > 0:
            image_norm = ((tds_image - np.min(tds_image)) / 
                         (np.max(tds_image) - np.min(tds_image)) * 255).astype(np.uint8)
        else:
            image_norm = np.zeros_like(tds_image, dtype=np.uint8)
        
        # Compute LBP
        lbp = local_binary_pattern(image_norm, n_points, radius, method='uniform')
        
        # Compute LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                              range=(0, n_points + 2), density=True)
        
        # LBP features
        for i, val in enumerate(hist):
            features[f'lbp_bin_{i}'] = val
        
        # LBP statistics
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_var'] = np.var(lbp)
        features['lbp_entropy'] = entropy(hist + 1e-10)
        
        return features
    
    def extract_gabor_features(self, tds_image: np.ndarray) -> Dict[str, float]:
        """
        Extract Gabor filter features from TDS image.
        
        Args:
            tds_image: Time-Doppler Spectrum image
            
        Returns:
            Dictionary of Gabor features
        """
        features = {}
        
        # Normalize image
        if np.max(tds_image) > 0:
            image_norm = (tds_image - np.min(tds_image)) / (np.max(tds_image) - np.min(tds_image))
        else:
            image_norm = np.zeros_like(tds_image)
        
        # Different frequencies and orientations
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for i, freq in enumerate(frequencies):
            for j, theta in enumerate(orientations):
                try:
                    # Apply Gabor filter
                    real, _ = gabor(image_norm, frequency=freq, theta=theta)
                    
                    # Extract features from filtered image
                    features[f'gabor_f{i}_t{j}_mean'] = np.mean(real)
                    features[f'gabor_f{i}_t{j}_var'] = np.var(real)
                    features[f'gabor_f{i}_t{j}_energy'] = np.sum(real ** 2)
                    
                except Exception:
                    # Handle edge cases
                    features[f'gabor_f{i}_t{j}_mean'] = 0
                    features[f'gabor_f{i}_t{j}_var'] = 0
                    features[f'gabor_f{i}_t{j}_energy'] = 0
        
        return features
    
    def extract_stft_features(self, iq_data: np.ndarray, 
                            window_size: int = 256) -> Dict[str, float]:
        """
        Extract features from Short-Time Fourier Transform.
        
        Args:
            iq_data: Complex I/Q data
            window_size: STFT window size
            
        Returns:
            Dictionary of STFT features
        """
        features = {}
        
        # Compute STFT
        frequencies, times, stft = signal.stft(iq_data, fs=self.fs, nperseg=window_size)
        stft_magnitude = np.abs(stft)
        
        # Features from STFT magnitude
        features['stft_mean'] = np.mean(stft_magnitude)
        features['stft_var'] = np.var(stft_magnitude)
        features['stft_max'] = np.max(stft_magnitude)
        features['stft_entropy'] = entropy(stft_magnitude.flatten() + 1e-10)
        
        # Temporal variation of spectral features
        spectral_centroids = []
        for t_idx in range(stft_magnitude.shape[1]):
            spectrum = stft_magnitude[:, t_idx]
            spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
            centroid = np.sum(frequencies * spectrum_norm)
            spectral_centroids.append(centroid)
        
        spectral_centroids = np.array(spectral_centroids)
        features['stft_centroid_mean'] = np.mean(spectral_centroids)
        features['stft_centroid_var'] = np.var(spectral_centroids)
        features['stft_centroid_range'] = np.max(spectral_centroids) - np.min(spectral_centroids)
        
        return features
    
    def extract_fractal_features(self, iq_data: np.ndarray) -> Dict[str, float]:
        """
        Extract fractal dimension features.
        
        Args:
            iq_data: Complex I/Q data
            
        Returns:
            Dictionary of fractal features
        """
        features = {}
        
        magnitude = np.abs(iq_data)
        
        # Higuchi fractal dimension
        def higuchi_fd(signal, k_max=10):
            N = len(signal)
            L = []
            x = []
            for k in range(1, k_max + 1):
                Lk = []
                for m in range(k):
                    Lmk = 0
                    for i in range(1, int((N - m) / k)):
                        Lmk += abs(signal[m + i * k] - signal[m + (i - 1) * k])
                    Lmk = Lmk * (N - 1) / (int((N - m) / k) * k) / k
                    Lmk = Lmk / k
                    Lk.append(Lmk)
                L.append(np.mean(Lmk))
                x.append(np.log(1.0 / k))
            
            # Linear regression to find slope (fractal dimension)
            if len(L) > 1:
                slope, _, _, _, _ = stats.linregress(x, np.log(L))
                return slope
            else:
                return 0
        
        try:
            features['fractal_dimension'] = higuchi_fd(magnitude)
        except Exception:
            features['fractal_dimension'] = 0
        
        return features
    
    def extract_complexity_features(self, iq_data: np.ndarray) -> Dict[str, float]:
        """
        Extract complexity measures.
        
        Args:
            iq_data: Complex I/Q data
            
        Returns:
            Dictionary of complexity features
        """
        features = {}
        
        magnitude = np.abs(iq_data)
        
        # Approximate entropy
        def approximate_entropy(data, m=2, r=None):
            if r is None:
                r = 0.2 * np.std(data)
            
            def _maxdist(xi, xj, N, m):
                return max([abs(ua - va) for ua, va in zip(xi, xj)])
            
            def _phi(m):
                patterns = np.array([data[i:i + m] for i in range(len(data) - m + 1)])
                C = np.zeros(len(patterns))
                for i in range(len(patterns)):
                    template_i = patterns[i]
                    for j in range(len(patterns)):
                        if _maxdist(template_i, patterns[j], len(patterns), m) <= r:
                            C[i] += 1.0
                phi = np.mean(np.log(C / len(patterns)))
                return phi
            
            return _phi(m) - _phi(m + 1)
        
        try:
            features['approximate_entropy'] = approximate_entropy(magnitude)
        except Exception:
            features['approximate_entropy'] = 0
        
        # Sample entropy (simplified version)
        def sample_entropy(data, m=2, r=None):
            if r is None:
                r = 0.2 * np.std(data)
            
            def _get_matches(data, m):
                patterns = [data[i:i + m] for i in range(len(data) - m + 1)]
                matches = 0
                for i in range(len(patterns)):
                    for j in range(i + 1, len(patterns)):
                        if max(abs(a - b) for a, b in zip(patterns[i], patterns[j])) <= r:
                            matches += 1
                return matches
            
            matches_m = _get_matches(data, m)
            matches_m1 = _get_matches(data, m + 1)
            
            if matches_m == 0 or matches_m1 == 0:
                return 0
            else:
                return -np.log(matches_m1 / matches_m)
        
        try:
            features['sample_entropy'] = sample_entropy(magnitude)
        except Exception:
            features['sample_entropy'] = 0
        
        return features
    
    def extract_all_features(self, iq_data: np.ndarray, 
                           spectrum: np.ndarray = None, 
                           frequencies: np.ndarray = None,
                           tds_image: np.ndarray = None) -> Dict[str, float]:
        """
        Extract all features from I/Q data.
        
        Args:
            iq_data: Complex I/Q data
            spectrum: Doppler spectrum (optional)
            frequencies: Frequency array (optional)
            tds_image: Time-Doppler Spectrum image (optional)
            
        Returns:
            Dictionary containing all features
        """
        all_features = {}
        
        # Temporal features
        temporal_features = self.extract_temporal_features(iq_data)
        all_features.update(temporal_features)
        
        # Spectral features (if spectrum provided)
        if spectrum is not None and frequencies is not None:
            spectral_features = self.extract_spectral_features(spectrum, frequencies)
            all_features.update(spectral_features)
        
        # STFT features
        stft_features = self.extract_stft_features(iq_data)
        all_features.update(stft_features)
        
        # Fractal features
        fractal_features = self.extract_fractal_features(iq_data)
        all_features.update(fractal_features)
        
        # Complexity features
        complexity_features = self.extract_complexity_features(iq_data)
        all_features.update(complexity_features)
        
        # TDS-based features (if TDS image provided)
        if tds_image is not None:
            lbp_features = self.extract_lbp_features(tds_image)
            all_features.update(lbp_features)
            
            gabor_features = self.extract_gabor_features(tds_image)
            all_features.update(gabor_features)
        
        return all_features
    
    def normalize_features(self, features_dict: Dict[str, float], 
                         fit_scaler: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Args:
            features_dict: Dictionary of features
            fit_scaler: Whether to fit the scaler
            
        Returns:
            Normalized feature array
        """
        feature_values = np.array(list(features_dict.values())).reshape(1, -1)
        
        # Replace any NaN or inf values
        feature_values = np.nan_to_num(feature_values, 
                                     nan=0.0, posinf=1e6, neginf=-1e6)
        
        if fit_scaler:
            return self.scaler.fit_transform(feature_values).flatten()
        else:
            return self.scaler.transform(feature_values).flatten()
    
    def get_feature_names(self, include_spectral: bool = True, 
                         include_tds: bool = True) -> List[str]:
        """
        Get list of all feature names.
        
        Args:
            include_spectral: Whether to include spectral features
            include_tds: Whether to include TDS-based features
            
        Returns:
            List of feature names
        """
        feature_names = []
        
        # Temporal features
        temporal_names = [
            'mag_mean', 'mag_var', 'mag_std', 'mag_skewness', 'mag_kurtosis',
            'mag_max', 'mag_min', 'phase_var', 'phase_std',
            'inst_freq_mean', 'inst_freq_var', 'inst_freq_std',
            'zcr_magnitude', 'zcr_real', 'zcr_imag',
            'energy', 'power', 'papr', 'crest_factor',
            'autocorr_first_zero', 'autocorr_lag1'
        ]
        feature_names.extend(temporal_names)
        
        # Spectral features
        if include_spectral:
            spectral_names = [
                'mean_magnitude', 'var_magnitude', 'std_magnitude',
                'max_magnitude', 'min_magnitude', 'spectral_centroid',
                'spectral_spread', 'spectral_skewness', 'spectral_kurtosis',
                'spectral_entropy', 'spectral_flatness', 'peak_frequency',
                'bandwidth_90', 'bandwidth_50', 'num_peaks',
                'spectral_rolloff', 'spectral_zcr'
            ]
            feature_names.extend(spectral_names)
        
        # STFT features
        stft_names = [
            'stft_mean', 'stft_var', 'stft_max', 'stft_entropy',
            'stft_centroid_mean', 'stft_centroid_var', 'stft_centroid_range'
        ]
        feature_names.extend(stft_names)
        
        # Fractal features
        feature_names.append('fractal_dimension')
        
        # Complexity features
        complexity_names = ['approximate_entropy', 'sample_entropy']
        feature_names.extend(complexity_names)
        
        # TDS features
        if include_tds:
            # LBP features (8 points + 2 = 10 bins)
            lbp_names = [f'lbp_bin_{i}' for i in range(10)]
            lbp_names.extend(['lbp_mean', 'lbp_var', 'lbp_entropy'])
            feature_names.extend(lbp_names)
            
            # Gabor features (3 frequencies × 4 orientations × 3 features)
            for i in range(3):
                for j in range(4):
                    feature_names.extend([
                        f'gabor_f{i}_t{j}_mean',
                        f'gabor_f{i}_t{j}_var',
                        f'gabor_f{i}_t{j}_energy'
                    ])
        
        return feature_names


if __name__ == "__main__":
    # Example usage
    
    # Create synthetic I/Q signal
    fs = 1000
    t = np.arange(2048) / fs
    iq_signal = np.exp(1j * 2 * np.pi * 50 * t) + 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    
    # Initialize feature extractor
    extractor = RadarFeatureExtractor(fs)
    
    # Extract features
    features = extractor.extract_all_features(iq_signal)
    
    print(f"Extracted {len(features)} features:")
    for name, value in list(features.items())[:10]:  # Show first 10 features
        print(f"  {name}: {value:.4f}")
    
    print("Feature extraction example complete!")