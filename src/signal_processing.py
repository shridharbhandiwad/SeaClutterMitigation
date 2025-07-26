"""
Signal processing utilities for radar I/Q data.
Includes windowing, Doppler spectrum computation, and TDS analysis.
"""

import numpy as np
import scipy.signal
from scipy import fft
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt


class RadarSignalProcessor:
    """
    Signal processing class for radar I/Q data analysis.
    Handles windowing, FFT, Doppler spectrum, and TDS computation.
    """
    
    def __init__(self, sampling_frequency: float = 1000.0):
        """
        Initialize the signal processor.
        
        Args:
            sampling_frequency: Sampling frequency in Hz
        """
        self.fs = sampling_frequency
    
    def segment_time_series(self, iq_data: np.ndarray, 
                           window_size: int = 1024, 
                           overlap_ratio: float = 0.5) -> Tuple[np.ndarray, List[int]]:
        """
        Segment I/Q time series into overlapping windows.
        
        Args:
            iq_data: Complex I/Q data (range_cells, time_samples)
            window_size: Size of each window
            overlap_ratio: Overlap ratio (0.0 to 1.0)
            
        Returns:
            Segmented data (range_cells, n_windows, window_size) and window indices
        """
        if iq_data.ndim == 1:
            iq_data = iq_data.reshape(1, -1)
        
        n_range_cells, n_time_samples = iq_data.shape
        
        # Calculate step size
        step_size = int(window_size * (1 - overlap_ratio))
        
        # Calculate number of windows
        n_windows = (n_time_samples - window_size) // step_size + 1
        
        # Create segmented data array
        segmented_data = np.zeros((n_range_cells, n_windows, window_size), dtype=complex)
        window_indices = []
        
        for i in range(n_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx <= n_time_samples:
                segmented_data[:, i, :] = iq_data[:, start_idx:end_idx]
                window_indices.append((start_idx, end_idx))
        
        return segmented_data[:, :len(window_indices), :], window_indices
    
    def apply_window_function(self, data: np.ndarray, 
                            window_type: str = 'hann') -> np.ndarray:
        """
        Apply window function to reduce spectral leakage.
        
        Args:
            data: Input data (..., window_size)
            window_type: Type of window ('hann', 'hamming', 'blackman', 'kaiser')
            
        Returns:
            Windowed data
        """
        window_size = data.shape[-1]
        
        if window_type == 'hann':
            window = np.hanning(window_size)
        elif window_type == 'hamming':
            window = np.hamming(window_size)
        elif window_type == 'blackman':
            window = np.blackman(window_size)
        elif window_type == 'kaiser':
            window = np.kaiser(window_size, beta=8.6)
        else:
            window = np.ones(window_size)  # Rectangular window
        
        # Reshape window for broadcasting
        window_shape = [1] * data.ndim
        window_shape[-1] = window_size
        window = window.reshape(window_shape)
        
        return data * window
    
    def compute_doppler_spectrum(self, iq_data: np.ndarray, 
                               window_type: str = 'hann',
                               zero_padding_factor: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Doppler spectrum using FFT.
        
        Args:
            iq_data: Complex I/Q data (..., time_samples)
            window_type: Window function type
            zero_padding_factor: Factor for zero padding (1 = no padding)
            
        Returns:
            Doppler spectrum magnitude and frequency array
        """
        # Apply window function
        windowed_data = self.apply_window_function(iq_data, window_type)
        
        # Zero padding
        if zero_padding_factor > 1:
            pad_length = (zero_padding_factor - 1) * iq_data.shape[-1]
            windowed_data = np.pad(windowed_data, 
                                 [(0, 0)] * (windowed_data.ndim - 1) + [(0, pad_length)],
                                 mode='constant')
        
        # Compute FFT
        spectrum = fft.fft(windowed_data, axis=-1)
        
        # Shift zero frequency to center
        spectrum = fft.fftshift(spectrum, axes=-1)
        
        # Compute magnitude
        magnitude_spectrum = np.abs(spectrum)
        
        # Create frequency array
        n_fft = spectrum.shape[-1]
        frequencies = fft.fftshift(fft.fftfreq(n_fft, 1/self.fs))
        
        return magnitude_spectrum, frequencies
    
    def compute_time_doppler_spectrum(self, iq_data: np.ndarray,
                                    window_size: int = 256,
                                    overlap_ratio: float = 0.75,
                                    window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Time-Doppler Spectrum (TDS) using STFT.
        
        Args:
            iq_data: Complex I/Q data (time_samples,)
            window_size: STFT window size
            overlap_ratio: Window overlap ratio
            window_type: Window function type
            
        Returns:
            TDS magnitude, time array, frequency array
        """
        if iq_data.ndim > 1:
            iq_data = iq_data.flatten()
        
        # Compute STFT
        nperseg = window_size
        noverlap = int(window_size * overlap_ratio)
        
        frequencies, times, stft = scipy.signal.stft(
            iq_data, 
            fs=self.fs,
            window=window_type,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=False
        )
        
        # Shift frequencies to center zero frequency
        frequencies = fft.fftshift(frequencies)
        stft = fft.fftshift(stft, axes=0)
        
        # Compute magnitude
        tds_magnitude = np.abs(stft)
        
        return tds_magnitude, times, frequencies
    
    def compute_power_spectral_density(self, iq_data: np.ndarray,
                                     window_type: str = 'hann',
                                     nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.
        
        Args:
            iq_data: Complex I/Q data
            window_type: Window function type
            nperseg: Length of each segment
            
        Returns:
            PSD and frequency array
        """
        if iq_data.ndim > 1:
            iq_data = iq_data.flatten()
        
        if nperseg is None:
            nperseg = min(256, len(iq_data) // 8)
        
        frequencies, psd = scipy.signal.welch(
            iq_data,
            fs=self.fs,
            window=window_type,
            nperseg=nperseg,
            return_onesided=False
        )
        
        # Shift to center zero frequency
        frequencies = fft.fftshift(frequencies)
        psd = fft.fftshift(psd)
        
        return psd, frequencies
    
    def compute_instantaneous_frequency(self, iq_data: np.ndarray) -> np.ndarray:
        """
        Compute instantaneous frequency using analytic signal.
        
        Args:
            iq_data: Complex I/Q data
            
        Returns:
            Instantaneous frequency
        """
        # Compute phase
        phase = np.unwrap(np.angle(iq_data))
        
        # Compute instantaneous frequency
        inst_freq = np.diff(phase) * self.fs / (2 * np.pi)
        
        return inst_freq
    
    def detect_chirp_signals(self, iq_data: np.ndarray,
                           method: str = 'stft') -> Dict:
        """
        Detect chirp signals (frequency modulated signals) in I/Q data.
        
        Args:
            iq_data: Complex I/Q data
            method: Detection method ('stft', 'wvd', 'instantaneous')
            
        Returns:
            Dictionary containing chirp detection results
        """
        results = {}
        
        if method == 'stft':
            # Use STFT to detect frequency changes over time
            tds, times, frequencies = self.compute_time_doppler_spectrum(iq_data)
            
            # Detect chirp by looking for diagonal patterns in TDS
            results['tds'] = tds
            results['times'] = times
            results['frequencies'] = frequencies
            
        elif method == 'instantaneous':
            # Use instantaneous frequency
            inst_freq = self.compute_instantaneous_frequency(iq_data)
            
            # Detect linear frequency change (chirp)
            freq_gradient = np.gradient(inst_freq)
            chirp_strength = np.std(freq_gradient)
            
            results['instantaneous_frequency'] = inst_freq
            results['frequency_gradient'] = freq_gradient
            results['chirp_strength'] = chirp_strength
        
        return results
    
    def remove_clutter(self, iq_data: np.ndarray, 
                      method: str = 'highpass',
                      cutoff_freq: float = 1.0) -> np.ndarray:
        """
        Remove low-frequency clutter from I/Q data.
        
        Args:
            iq_data: Complex I/Q data
            method: Clutter removal method ('highpass', 'moving_average')
            cutoff_freq: Cutoff frequency for filtering
            
        Returns:
            Clutter-filtered data
        """
        if method == 'highpass':
            # Design high-pass filter
            nyquist = self.fs / 2
            normalized_cutoff = cutoff_freq / nyquist
            
            b, a = scipy.signal.butter(4, normalized_cutoff, btype='high')
            
            # Apply filter to real and imaginary parts
            filtered_real = scipy.signal.filtfilt(b, a, np.real(iq_data))
            filtered_imag = scipy.signal.filtfilt(b, a, np.imag(iq_data))
            
            filtered_data = filtered_real + 1j * filtered_imag
            
        elif method == 'moving_average':
            # Remove moving average (low-pass component)
            window_size = int(self.fs / cutoff_freq)
            
            # Compute moving average
            kernel = np.ones(window_size) / window_size
            moving_avg_real = np.convolve(np.real(iq_data), kernel, mode='same')
            moving_avg_imag = np.convolve(np.imag(iq_data), kernel, mode='same')
            moving_avg = moving_avg_real + 1j * moving_avg_imag
            
            # Subtract moving average
            filtered_data = iq_data - moving_avg
        
        else:
            filtered_data = iq_data
        
        return filtered_data


def create_complex_iq_from_real(i_data: np.ndarray, q_data: np.ndarray) -> np.ndarray:
    """
    Create complex I/Q data from separate I and Q arrays.
    
    Args:
        i_data: In-phase component
        q_data: Quadrature component
        
    Returns:
        Complex I/Q data
    """
    return i_data + 1j * q_data


def compute_magnitude_phase(iq_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute magnitude and phase from complex I/Q data.
    
    Args:
        iq_data: Complex I/Q data
        
    Returns:
        Magnitude and phase arrays
    """
    magnitude = np.abs(iq_data)
    phase = np.angle(iq_data)
    
    return magnitude, phase


if __name__ == "__main__":
    # Example usage
    
    # Create synthetic I/Q signal
    fs = 1000  # Sampling frequency
    t = np.arange(2048) / fs
    
    # Target signal with Doppler shift
    target_freq = 50
    target_signal = np.exp(1j * 2 * np.pi * target_freq * t)
    
    # Add noise
    noise = 0.1 * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
    iq_signal = target_signal + noise
    
    # Initialize processor
    processor = RadarSignalProcessor(fs)
    
    # Segment the data
    segmented_data, indices = processor.segment_time_series(iq_signal, window_size=512, overlap_ratio=0.5)
    print(f"Segmented data shape: {segmented_data.shape}")
    
    # Compute Doppler spectrum
    spectrum, frequencies = processor.compute_doppler_spectrum(iq_signal)
    print(f"Spectrum shape: {spectrum.shape}")
    
    # Compute TDS
    tds, times, freqs = processor.compute_time_doppler_spectrum(iq_signal)
    print(f"TDS shape: {tds.shape}")
    
    print("Signal processing example complete!")