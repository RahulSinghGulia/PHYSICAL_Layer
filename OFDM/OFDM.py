import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import interpolate
from scipy import signal
import os
import argparse
from typing import Dict, List, Tuple, Union, Optional
import time

import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg', 'GTK3Agg' depending on your system

class ModulationScheme:
    """Class to handle different modulation schemes"""
    
    @staticmethod
    def get_config(mod_scheme: str) -> Dict:
        """
        Get configuration for a specific modulation scheme
        
        Args:
            mod_scheme: Name of the modulation scheme
            
        Returns:
            Dictionary with bits_per_symbol and constellation points
        """
        if mod_scheme == 'BPSK':
            return {'bits_per_symbol': 1, 'constellation': np.array([-1+0j, 1+0j])}
        elif mod_scheme == 'QPSK':
            return {'bits_per_symbol': 2, 'constellation': np.array([-1-1j, -1+1j, 1-1j, 1+1j]) / np.sqrt(2)}
        elif mod_scheme == '8-PSK':
            return {'bits_per_symbol': 3, 'constellation': np.exp(1j * np.arange(0, 2 * np.pi, 2 * np.pi / 8))}
        elif mod_scheme == '16-QAM':
            return {'bits_per_symbol': 4, 'constellation': np.array([-3-3j, -3-1j, -3+3j, -3+1j,
                                                                    -1-3j, -1-1j, -1+3j, -1+1j,
                                                                    3-3j,  3-1j,  3+3j,  3+1j,
                                                                    1-3j,  1-1j,  1+3j,  1+1j]) / np.sqrt(10)}
        elif mod_scheme == '64-QAM':
            levels = [-7, -5, -3, -1, 1, 3, 5, 7]
            constellation = np.array([x + 1j * y for y in levels for x in levels]) / np.sqrt(42)
            return {'bits_per_symbol': 6, 'constellation': constellation}
        elif mod_scheme == '256-QAM':
            levels = [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15]
            constellation = np.array([x + 1j * y for y in levels for x in levels]) / np.sqrt(170)
            return {'bits_per_symbol': 8, 'constellation': constellation}
        elif mod_scheme == '1024-QAM':  # Added higher order modulation
            levels = np.arange(-31, 32, 2)
            constellation = np.array([x + 1j * y for y in levels for x in levels]) / np.sqrt(682)
            return {'bits_per_symbol': 10, 'constellation': constellation}
        else:
            raise ValueError(f"Unsupported modulation scheme: {mod_scheme}")
    
    @staticmethod
    def modulate(bits: np.ndarray, mod_scheme: str) -> np.ndarray:
        """
        Convert bits to complex symbols according to modulation scheme
        
        Args:
            bits: Binary data array
            mod_scheme: Modulation scheme name
            
        Returns:
            Complex symbols
        """
        config = ModulationScheme.get_config(mod_scheme)
        bits_per_symbol = config['bits_per_symbol']
        constellation = config['constellation']
        
        # Ensure bits length is multiple of bits_per_symbol
        num_symbols = len(bits) // bits_per_symbol
        if len(bits) % bits_per_symbol != 0:
            print(f"Warning: {len(bits) % bits_per_symbol} bits truncated to ensure integer number of symbols")
            bits = bits[:num_symbols * bits_per_symbol]
        
        # Convert bit groups to indices
        indices = bits.reshape(-1, bits_per_symbol).dot(1 << np.arange(bits_per_symbol - 1, -1, -1))
        return constellation[indices]
    
    @staticmethod
    def demodulate(symbols: np.ndarray, mod_scheme: str) -> np.ndarray:
        """
        Convert complex symbols back to bits
        
        Args:
            symbols: Complex symbols
            mod_scheme: Modulation scheme name
            
        Returns:
            Binary data array
        """
        config = ModulationScheme.get_config(mod_scheme)
        constellation = config['constellation']
        bits_per_symbol = config['bits_per_symbol']
        
        # For each symbol, find closest constellation point
        demodulated_bits = []
        for symbol in symbols:
            distances = np.abs(constellation - symbol)**2
            closest_index = np.argmin(distances)
            binary_representation = np.binary_repr(closest_index, width=bits_per_symbol)
            demodulated_bits.extend([int(b) for b in binary_representation])
            
        return np.array(demodulated_bits)


class ChannelModel:
    """Class to handle different wireless channel models"""
    
    @staticmethod
    def create_multipath_channel(num_taps: int = 4, max_delay_spread: float = 2.0, 
                                sampling_rate: float = 1.0, K_factor: float = 10.0) -> np.ndarray:
        """
        Create a realistic multipath channel with specified parameters
        
        Args:
            num_taps: Number of multipath components
            max_delay_spread: Maximum delay spread in samples
            sampling_rate: Sampling rate
            K_factor: Rician K-factor for LOS vs NLOS energy
            
        Returns:
            Complex channel taps
        """
        # Generate tap delays - exponentially distributed
        delays = np.sort(np.random.exponential(scale=max_delay_spread/2, size=num_taps-1))
        delays = np.insert(delays, 0, 0)  # First tap at delay 0
        
        # Generate tap powers - exponential power decay
        avg_power_db = -delays / max_delay_spread * 20  # 20 dB decay over max delay
        avg_power_linear = 10 ** (avg_power_db / 10)
        
        # Add Rician fading to first tap (LOS), Rayleigh to others (NLOS)
        channel_taps = np.zeros(num_taps, dtype=complex)
        
        # LOS component for first tap
        los_component = np.sqrt(K_factor / (K_factor + 1)) * avg_power_linear[0]
        nlos_component = np.sqrt(1 / (K_factor + 1)) * avg_power_linear[0]
        channel_taps[0] = los_component + nlos_component * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        
        # NLOS components for other taps
        for i in range(1, num_taps):
            channel_taps[i] = avg_power_linear[i] * (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        
        # Normalize channel energy
        channel_taps = channel_taps / np.sqrt(np.sum(np.abs(channel_taps)**2))
        
        # Convert to time-domain filter
        filter_length = int(np.ceil(delays[-1]) + 1)
        h = np.zeros(filter_length, dtype=complex)
        for i, delay in enumerate(delays):
            idx = int(np.floor(delay))
            h[idx] += channel_taps[i]
        
        return h
    
    @staticmethod
    def add_cfo(signal: np.ndarray, cfo_hz: float, sampling_rate: float) -> np.ndarray:
        """
        Add carrier frequency offset to signal
        
        Args:
            signal: Complex baseband signal
            cfo_hz: Carrier frequency offset in Hz
            sampling_rate: Sampling rate in Hz
            
        Returns:
            Signal with CFO
        """
        t = np.arange(len(signal)) / sampling_rate
        cfo_factor = np.exp(1j * 2 * np.pi * cfo_hz * t)
        return signal * cfo_factor
    
    @staticmethod
    def add_sfo(signal: np.ndarray, sfo_ppm: float) -> np.ndarray:
        """
        Add sampling frequency offset to signal
        
        Args:
            signal: Complex baseband signal
            sfo_ppm: Sampling frequency offset in parts per million
            
        Returns:
            Signal with SFO
        """
        # Convert ppm to ratio
        sfo_ratio = sfo_ppm / 1e6
        
        # Create new time indices for interpolation
        original_indices = np.arange(len(signal))
        new_indices = np.arange(0, len(signal), 1 + sfo_ratio)
        new_indices = new_indices[new_indices < len(signal)]
        
        # Interpolate signal
        real_interp = interpolate.interp1d(original_indices, np.real(signal), bounds_error=False, fill_value=0)
        imag_interp = interpolate.interp1d(original_indices, np.imag(signal), bounds_error=False, fill_value=0)
        
        # Return interpolated signal padded to original length
        resampled = real_interp(new_indices) + 1j * imag_interp(new_indices)
        return np.pad(resampled, (0, len(signal) - len(resampled)), mode='constant')
    
    @staticmethod
    def apply_channel(signal: np.ndarray, channel_taps: np.ndarray, snr_db: float) -> np.ndarray:
        """
        Apply channel effects (multipath, noise) to signal
        
        Args:
            signal: Complex baseband signal
            channel_taps: Complex channel taps
            snr_db: Signal-to-noise ratio in dB
            
        Returns:
            Signal after channel effects
        """
        # Apply multipath channel
        signal_faded = np.convolve(signal, channel_taps, mode='same')
        
        # Add noise
        signal_power = np.mean(np.abs(signal_faded)**2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (np.random.randn(len(signal_faded)) + 
                                           1j * np.random.randn(len(signal_faded)))
        
        return signal_faded + noise


class ErrorCorrection:
    """Class to handle error correction coding"""
    
    @staticmethod
    def repetition_encode(bits: np.ndarray, repetitions: int = 3) -> np.ndarray:
        """
        Simple repetition code
        
        Args:
            bits: Input bits
            repetitions: Number of repetitions
            
        Returns:
            Encoded bits
        """
        return np.repeat(bits, repetitions)
    
    @staticmethod
    def repetition_decode(encoded_bits: np.ndarray, repetitions: int = 3) -> np.ndarray:
        """
        Decode repetition code using majority vote
        
        Args:
            encoded_bits: Encoded bits
            repetitions: Number of repetitions
            
        Returns:
            Decoded bits
        """
        # Ensure length is multiple of repetitions
        truncated_length = (len(encoded_bits) // repetitions) * repetitions
        reshaped = encoded_bits[:truncated_length].reshape(-1, repetitions)
        return np.round(np.mean(reshaped, axis=1)).astype(int)
    
    @staticmethod
    def hamming_encode(bits: np.ndarray) -> np.ndarray:
        """
        Hamming(7,4) encoding
        
        Args:
            bits: Input bits (length should be multiple of 4)
            
        Returns:
            Encoded bits
        """
        # Ensure length is multiple of 4
        num_blocks = len(bits) // 4
        if len(bits) % 4 != 0:
            print(f"Warning: {len(bits) % 4} bits truncated for Hamming encoding")
            bits = bits[:num_blocks * 4]
        
        # Generator matrix for Hamming(7,4)
        G = np.array([
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Reshape input to blocks of 4 bits
        input_blocks = bits.reshape(-1, 4)
        encoded_blocks = np.zeros((num_blocks, 7), dtype=int)
        
        # Apply generator matrix to each block
        for i in range(num_blocks):
            encoded_blocks[i] = np.remainder(G.dot(input_blocks[i]), 2)
            
        return encoded_blocks.flatten()
    
    @staticmethod
    def hamming_decode(encoded_bits: np.ndarray) -> np.ndarray:
        """
        Hamming(7,4) decoding
        
        Args:
            encoded_bits: Encoded bits
            
        Returns:
            Decoded bits
        """
        # Ensure length is multiple of 7
        num_blocks = len(encoded_bits) // 7
        if len(encoded_bits) % 7 != 0:
            print(f"Warning: {len(encoded_bits) % 7} bits truncated for Hamming decoding")
            encoded_bits = encoded_bits[:num_blocks * 7]
        
        # Parity check matrix for Hamming(7,4)
        H = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1]
        ])
        
        # Reshape input to blocks of 7 bits
        encoded_blocks = encoded_bits.reshape(-1, 7)
        decoded_blocks = np.zeros((num_blocks, 4), dtype=int)
        
        # Process each block
        for i in range(num_blocks):
            # Calculate syndrome
            syndrome = np.remainder(H.dot(encoded_blocks[i]), 2)
            
            # Convert syndrome to decimal for error location
            error_loc = syndrome[0] * 1 + syndrome[1] * 2 + syndrome[2] * 4
            
            # Correct error if necessary
            corrected_block = encoded_blocks[i].copy()
            if error_loc > 0:
                corrected_block[error_loc - 1] = 1 - corrected_block[error_loc - 1]
            
            # Extract data bits (positions 2, 4, 5, 6 in 0-indexed array)
            decoded_blocks[i] = corrected_block[[2, 4, 5, 6]]
            
        return decoded_blocks.flatten()


class OFDMSystem:
    """Main OFDM system class"""
    
    def __init__(self, num_subcarriers: int = 64, cp_length: int = 16, 
                 modulation: str = 'QPSK', pilot_spacing: int = 8,
                 coding_scheme: str = 'none', papr_reduction: str = 'none'):
        """
        Initialize OFDM system
        
        Args:
            num_subcarriers: Number of OFDM subcarriers
            cp_length: Cyclic prefix length
            modulation: Modulation scheme
            pilot_spacing: Distance between pilot subcarriers
            coding_scheme: Error correction coding scheme ('none', 'repetition', 'hamming')
            papr_reduction: PAPR reduction technique ('none', 'clipping', 'tone_reservation')
        """
        self.N = num_subcarriers
        self.CP = cp_length
        self.modulation = modulation
        self.pilot_spacing = pilot_spacing
        self.coding_scheme = coding_scheme
        self.papr_reduction = papr_reduction
        
        # Initialize pilot positions
        self.pilot_positions = np.arange(0, self.N, pilot_spacing)
        self.pilot_value = 1 + 1j
        
        # Track active subcarriers (excluding DC and edge for guard bands)
        guard_band_size = self.N // 8
        self.active_indices = np.ones(self.N, dtype=bool)
        self.active_indices[0] = False  # DC null
        self.active_indices[:guard_band_size] = False  # Lower guard band
        self.active_indices[-guard_band_size:] = False  # Upper guard band
        
        # But pilots should be active regardless
        self.active_indices[self.pilot_positions] = True
        
        # Reserved tones for PAPR reduction (if used)
        if papr_reduction == 'tone_reservation':
            # Reserve ~5% of subcarriers for PAPR reduction
            num_reserved = max(1, int(0.05 * self.N))
            reserved_candidates = np.setdiff1d(np.where(self.active_indices)[0], self.pilot_positions)
            np.random.shuffle(reserved_candidates)
            self.reserved_tones = reserved_candidates[:num_reserved]
            self.active_indices[self.reserved_tones] = False
        else:
            self.reserved_tones = np.array([], dtype=int)
    
    def encode_bits(self, bits: np.ndarray) -> np.ndarray:
        """
        Apply error correction coding
        
        Args:
            bits: Input bits
            
        Returns:
            Encoded bits
        """
        if self.coding_scheme == 'none':
            return bits
        elif self.coding_scheme == 'repetition':
            return ErrorCorrection.repetition_encode(bits, 3)
        elif self.coding_scheme == 'hamming':
            return ErrorCorrection.hamming_encode(bits)
        else:
            raise ValueError(f"Unsupported coding scheme: {self.coding_scheme}")
    
    def decode_bits(self, bits: np.ndarray) -> np.ndarray:
        """
        Decode error correction coding
        
        Args:
            bits: Encoded bits
            
        Returns:
            Decoded bits
        """
        if self.coding_scheme == 'none':
            return bits
        elif self.coding_scheme == 'repetition':
            return ErrorCorrection.repetition_decode(bits, 3)
        elif self.coding_scheme == 'hamming':
            return ErrorCorrection.hamming_decode(bits)
        else:
            raise ValueError(f"Unsupported coding scheme: {self.coding_scheme}")
    
    def reduce_papr(self, time_blocks: np.ndarray) -> np.ndarray:
        """
        Apply PAPR reduction technique
        
        Args:
            time_blocks: OFDM time domain blocks
            
        Returns:
            Time domain blocks with reduced PAPR
        """
        if self.papr_reduction == 'none':
            return time_blocks
        
        elif self.papr_reduction == 'clipping':
            # Simple amplitude clipping
            clip_ratio = 1.5  # Clip at 1.5x average power
            avg_power = np.mean(np.abs(time_blocks)**2)
            clip_threshold = np.sqrt(clip_ratio * avg_power)
            
            # Get magnitudes
            magnitudes = np.abs(time_blocks)
            # Find indices where magnitude exceeds threshold
            clip_indices = magnitudes > clip_threshold
            # Clip while preserving phase
            if np.any(clip_indices):
                phases = np.angle(time_blocks[clip_indices])
                time_blocks[clip_indices] = clip_threshold * np.exp(1j * phases)
            
            return time_blocks
        
        elif self.papr_reduction == 'tone_reservation':
            # Use reserved tones to reduce PAPR
            if len(self.reserved_tones) == 0:
                return time_blocks
            
            # For each OFDM symbol, iterate to find good reserved tone values
            for i in range(time_blocks.shape[0]):
                freq_block = np.fft.fft(time_blocks[i])
                best_papr = self.calculate_papr(time_blocks[i])
                best_values = np.zeros(len(self.reserved_tones), dtype=complex)
                
                # Simple greedy search
                for _ in range(10):  # Try 10 random combinations
                    # Random complex values for reserved tones
                    reserved_values = (np.random.randn(len(self.reserved_tones)) + 
                                     1j * np.random.randn(len(self.reserved_tones)))
                    
                    # Test this combination
                    test_freq = freq_block.copy()
                    test_freq[self.reserved_tones] = reserved_values
                    test_time = np.fft.ifft(test_freq)
                    test_papr = self.calculate_papr(test_time)
                    
                    # Keep if better
                    if test_papr < best_papr:
                        best_papr = test_papr
                        best_values = reserved_values
                
                # Apply best values
                freq_block[self.reserved_tones] = best_values
                time_blocks[i] = np.fft.ifft(freq_block)
            
            return time_blocks
        
        else:
            raise ValueError(f"Unsupported PAPR reduction: {self.papr_reduction}")
    
    def calculate_papr(self, time_signal: np.ndarray) -> float:
        """Calculate Peak-to-Average Power Ratio"""
        peak_power = np.max(np.abs(time_signal)**2)
        avg_power = np.mean(np.abs(time_signal)**2)
        return peak_power / avg_power
    
    def transmit(self, bits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transmit bits through OFDM system
        
        Args:
            bits: Binary data to transmit
            
        Returns:
            Tuple of (transmitted signal, original symbols)
        """
        # Apply error correction coding if enabled
        coded_bits = self.encode_bits(bits)
        
        # Modulate bits to symbols
        symbols = ModulationScheme.modulate(coded_bits, self.modulation)
        original_symbols = symbols.copy()  # Keep copy for comparison
        
        # Calculate how many symbols we can fit per OFDM block (accounting for pilots and inactive)
        data_carriers = self.N - len(self.pilot_positions) - np.sum(~self.active_indices)
        
        # Pad symbols if needed
        total_needed_symbols = int(np.ceil(len(symbols) / data_carriers) * data_carriers)
        if len(symbols) < total_needed_symbols:
            symbols = np.pad(symbols, (0, total_needed_symbols - len(symbols)), mode='constant')
        
        # Reshape into blocks with proper mapping
        num_blocks = total_needed_symbols // data_carriers
        ofdm_blocks = np.zeros((num_blocks, self.N), dtype=complex)
        
        # Map symbols to subcarriers (excluding pilots and inactive)
        symbol_idx = 0
        for block_idx in range(num_blocks):
            for carrier_idx in range(self.N):
                if (carrier_idx not in self.pilot_positions and 
                    self.active_indices[carrier_idx] and
                    carrier_idx not in self.reserved_tones):
                    if symbol_idx < len(symbols):
                        ofdm_blocks[block_idx, carrier_idx] = symbols[symbol_idx]
                        symbol_idx += 1
            
            # Insert pilots
            ofdm_blocks[block_idx, self.pilot_positions] = self.pilot_value
        
        # Convert to time domain
        time_blocks = np.fft.ifft(ofdm_blocks, axis=1)
        
        # Apply PAPR reduction
        time_blocks = self.reduce_papr(time_blocks)
        
        # Add cyclic prefix
        time_blocks_cp = np.concatenate([time_blocks[:, -self.CP:], time_blocks], axis=1)
        
        # Flatten to create transmit signal
        tx_signal = time_blocks_cp.flatten()
        
        return tx_signal, original_symbols
    
    def synchronize(self, rx_signal: np.ndarray) -> np.ndarray:
        """
        Perform time and frequency synchronization
        
        Args:
            rx_signal: Received signal
            
        Returns:
            Synchronized signal
        """
        # For simplicity, we'll implement a basic correlation-based sync
        # In a real system, you'd use training sequences/preambles
        
        # Extract one OFDM symbol with CP as correlation reference
        symbol_len = self.N + self.CP
        if len(rx_signal) < symbol_len:
            return rx_signal  # Can't do sync with too short signal
            
        # Use correlation of CP with corresponding end of symbol
        correlation = np.zeros(len(rx_signal) - symbol_len, dtype=complex)
        
        for i in range(len(correlation)):
            cp_segment = rx_signal[i:i+self.CP]
            symbol_end = rx_signal[i+symbol_len-self.CP:i+symbol_len]
            correlation[i] = np.abs(np.sum(cp_segment * np.conj(symbol_end)))
        
        # Find peaks in correlation (OFDM symbol boundaries)
        # Using a simple peak finding approach
        if len(correlation) > 0:
            peak_idx = np.argmax(correlation)
            # Align to symbol boundary
            return rx_signal[peak_idx:]
        else:
            return rx_signal
    
    def estimate_cfo(self, rx_signal: np.ndarray) -> float:
        """
        Estimate carrier frequency offset
        
        Args:
            rx_signal: Received signal
            
        Returns:
            Estimated CFO normalized to subcarrier spacing
        """
        # Use CP correlation for CFO estimation
        symbol_len = self.N + self.CP
        num_symbols = len(rx_signal) // symbol_len
        
        if num_symbols < 1:
            return 0.0
        
        cfo_angles = []
        for i in range(min(num_symbols, 10)):  # Use up to 10 symbols
            start_idx = i * symbol_len
            cp_segment = rx_signal[start_idx:start_idx+self.CP]
            symbol_end = rx_signal[start_idx+symbol_len-self.CP:start_idx+symbol_len]
            
            # Phase of correlation gives CFO
            correlation = np.sum(cp_segment * np.conj(symbol_end))
            cfo_angles.append(np.angle(correlation))
        
        # Average and normalize by CP length and 2π
        if cfo_angles:
            avg_angle = np.mean(cfo_angles)
            return avg_angle / (2 * np.pi * self.CP)
        else:
            return 0.0
    
    def correct_cfo(self, rx_signal: np.ndarray, cfo_norm: float) -> np.ndarray:
        """
        Correct carrier frequency offset
        
        Args:
            rx_signal: Received signal
            cfo_norm: Normalized CFO estimate
            
        Returns:
            CFO-corrected signal
        """
        t = np.arange(len(rx_signal))
        correction = np.exp(-1j * 2 * np.pi * cfo_norm * t)
        return rx_signal * correction
    
    def receive(self, rx_signal: np.ndarray, channel_taps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """
        Receive and demodulate OFDM signal
        
        Args:
            rx_signal: Received signal
            channel_taps: Channel impulse response (if known)
            
        Returns:
            Tuple of (decoded bits, metrics dictionary)
        """
        metrics = {}
        start_time = time.time()
        
        # Perform synchronization
        rx_signal_sync = self.synchronize(rx_signal)
        metrics['sync_time'] = time.time() - start_time
        
        # Estimate and correct CFO
        cfo_estimate = self.estimate_cfo(rx_signal_sync)
        rx_signal_cfo_corrected = self.correct_cfo(rx_signal_sync, cfo_estimate)
        metrics['cfo_estimate'] = cfo_estimate
        
        # Determine number of complete OFDM symbols
        symbol_len = self.N + self.CP
        num_symbols = len(rx_signal_cfo_corrected) // symbol_len
        
        if num_symbols == 0:
            return np.array([]), metrics
        
        # Remove extra samples
        rx_signal_truncated = rx_signal_cfo_corrected[:num_symbols * symbol_len]
        
        # Reshape to extract OFDM symbols with CP
        rx_blocks = rx_signal_truncated.reshape(-1, symbol_len)
        
        # Remove CP
        rx_blocks_no_cp = rx_blocks[:, self.CP:]
        
        # FFT to frequency domain
        rx_symbols_blocks = np.fft.fft(rx_blocks_no_cp, axis=1)
        
        # Channel estimation using pilots
        H_est = np.zeros_like(rx_symbols_blocks, dtype=complex)
        for block_idx in range(rx_symbols_blocks.shape[0]):
            # Extract pilots
            received_pilots = rx_symbols_blocks[block_idx, self.pilot_positions]
            channel_at_pilots = received_pilots / self.pilot_value
            
            # Interpolate channel
            pilot_indices = np.array(self.pilot_positions)
            
            # Handle edge cases for interpolation
            if len(pilot_indices) < 2:
                # Can't interpolate with only one point, use it for all
                H_est[block_idx, :] = channel_at_pilots[0]
            else:
                # Linear interpolation for channel estimation
                interp_real = interpolate.interp1d(pilot_indices, np.real(channel_at_pilots), 
                                                 kind='linear', fill_value='extrapolate')
                interp_imag = interpolate.interp1d(pilot_indices, np.imag(channel_at_pilots), 
                                                 kind='linear', fill_value='extrapolate')
                
                # Create full channel estimate
                H_est[block_idx, :] = interp_real(np.arange(self.N)) + 1j * interp_imag(np.arange(self.N))
        
        # Channel equalization
        equalized_blocks = rx_symbols_blocks / H_est
        
        # Extract data symbols
        data_symbols = []
        for block_idx in range(equalized_blocks.shape[0]):
            # Get only the active subcarriers that aren't pilots or reserved tones
            active_mask = self.active_indices.copy()
            active_mask[self.pilot_positions] = False
            active_mask[self.reserved_tones] = False
            
            block_symbols = equalized_blocks[block_idx, active_mask]
            data_symbols.extend(block_symbols)
        
        data_symbols = np.array(data_symbols)
        
        # Demodulate symbols to bits
        demod_bits = ModulationScheme.demodulate(data_symbols, self.modulation)
        
        # Apply error correction decoding
        decoded_bits = self.decode_bits(demod_bits)
        
        # Calculate metrics
        metrics['num_symbols'] = num_symbols
        metrics['evm'] = self.calculate_evm(data_symbols, self.modulation)
        metrics['ber'] = 0  # Will be calculated later when comparing with original
        metrics['processing_time'] = time.time() - start_time
        
        return decoded_bits, metrics
    
    def calculate_evm(self, symbols: np.ndarray, mod_scheme: str) -> float:
        """
        Calculate Error Vector Magnitude (EVM)
        
        Args:
            symbols: Received symbols
            mod_scheme: Modulation scheme
            
        Returns:
            EVM as percentage
        """
        config = ModulationScheme.get_config(mod_scheme)
        constellation = config['constellation']
        
        # Find closest constellation points
        distances = np.abs(symbols[:, np.newaxis] - constellation)
        closest_indices = np.argmin(distances, axis=1)
        ideal_symbols = constellation[closest_indices]
        
        # Calculate EVM
        error = symbols - ideal_symbols
        evm_rms = np.sqrt(np.mean(np.abs(error)**2)) / np.sqrt(np.mean(np.abs(ideal_symbols)**2))
        return 100 * evm_rms
    
    def simulate(self, num_bits: int = 1000, snr_db: float = 20.0, 
                cfo_hz: float = 0.0, sfo_ppm: float = 0.0) -> Dict:
        """
        Complete simulation of OFDM system
        
        Args:
            num_bits: Number of bits to transmit
            snr_db: Signal-to-noise ratio in dB
            cfo_hz: Carrier frequency offset in Hz
            sfo_ppm: Sampling frequency offset in ppm
            
        Returns:
            Dictionary with simulation results and metrics
        """
        # Generate random bits
        tx_bits = np.random.randint(0, 2, num_bits)
        
        # Transmit
        tx_signal, original_symbols = self.transmit(tx_bits)
        
        # Create channel
        channel_taps = ChannelModel.create_multipath_channel()
        
        # Apply channel effects
        rx_signal = ChannelModel.apply_channel(tx_signal, channel_taps, snr_db)
        
        # Add impairments
        if cfo_hz != 0:
            rx_signal = ChannelModel.add_cfo(rx_signal, cfo_hz, 1.0)
        if sfo_ppm != 0:
            rx_signal = ChannelModel.add_sfo(rx_signal, sfo_ppm)
        
        # Receive
        rx_bits, metrics = self.receive(rx_signal)
        
        # Calculate BER (only on comparable length)
        min_length = min(len(tx_bits), len(rx_bits))
        if min_length > 0:
            errors = np.sum(tx_bits[:min_length] != rx_bits[:min_length])
            metrics['ber'] = errors / min_length
        else:
            metrics['ber'] = 1.0
        
        # Calculate PAPR
        metrics['papr'] = self.calculate_papr(tx_signal)
        
        return {
            'tx_bits': tx_bits,
            'rx_bits': rx_bits,
            'original_symbols': original_symbols,
            'metrics': metrics,
            'channel_taps': channel_taps,
            'tx_signal': tx_signal,
            'rx_signal': rx_signal
        }


def plot_constellation(symbols: np.ndarray, mod_scheme: str, title: str = ""):
    """
    Plot constellation diagram
    
    Args:
        symbols: Complex symbols
        mod_scheme: Modulation scheme
        title: Plot title
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(np.real(symbols), np.imag(symbols), alpha=0.5)
    
    # Plot reference constellation
    config = ModulationScheme.get_config(mod_scheme)
    ref_symbols = config['constellation']
    plt.scatter(np.real(ref_symbols), np.imag(ref_symbols), color='red', marker='x', label='Reference')
    
    plt.xlabel('In-phase')
    plt.ylabel('Quadrature')
    plt.title(title)
    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show(block=True)
    plt.savefig('constellation.png')
    plt.close()


def plot_spectrum(signal: np.ndarray, fs: float = 1.0, title: str = ""):
    """
    Plot power spectral density
    
    Args:
        signal: Time domain signal
        fs: Sampling frequency
        title: Plot title
    """
    plt.figure()
    freqs = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/fs))
    psd = np.fft.fftshift(np.abs(np.fft.fft(signal))**2)
    plt.plot(freqs, 10 * np.log10(psd))
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title(title)
    plt.grid()
    plt.show(block=True)
    plt.savefig('spectrum.png')
    plt.close()


def main():
    """Main function to demonstrate OFDM system"""
    parser = argparse.ArgumentParser(description='OFDM System Simulation')
    parser.add_argument('--num_bits', type=int, default=1000, help='Number of bits to transmit')
    parser.add_argument('--modulation', type=str, default='QPSK', 
                       choices=['BPSK', 'QPSK', '8-PSK', '16-QAM', '64-QAM', '256-QAM', '1024-QAM'],
                       help='Modulation scheme')
    parser.add_argument('--num_subcarriers', type=int, default=64, help='Number of OFDM subcarriers')
    parser.add_argument('--cp_length', type=int, default=16, help='Cyclic prefix length')
    parser.add_argument('--snr_db', type=float, default=20.0, help='Signal-to-noise ratio in dB')
    parser.add_argument('--cfo_hz', type=float, default=0.0, help='Carrier frequency offset in Hz')
    parser.add_argument('--sfo_ppm', type=float, default=0.0, help='Sampling frequency offset in ppm')
    parser.add_argument('--coding', type=str, default='none', 
                       choices=['none', 'repetition', 'hamming'], help='Error correction coding')
    parser.add_argument('--papr_reduction', type=str, default='none',
                       choices=['none', 'clipping', 'tone_reservation'], help='PAPR reduction technique')
    
    args = parser.parse_args()
    
    # Initialize OFDM system
    ofdm = OFDMSystem(
        num_subcarriers=args.num_subcarriers,
        cp_length=args.cp_length,
        modulation=args.modulation,
        coding_scheme=args.coding,
        papr_reduction=args.papr_reduction
    )
    
    # Run simulation
    results = ofdm.simulate(
        num_bits=args.num_bits,
        snr_db=args.snr_db,
        cfo_hz=args.cfo_hz,
        sfo_ppm=args.sfo_ppm
    )
    
    # Print results
    print("\nSimulation Results:")
    print(f"Modulation: {args.modulation}")
    print(f"Subcarriers: {args.num_subcarriers}")
    print(f"CP Length: {args.cp_length}")
    print(f"SNR: {args.snr_db} dB")
    print(f"CFO: {args.cfo_hz} Hz")
    print(f"SFO: {args.sfo_ppm} ppm")
    print(f"Coding: {args.coding}")
    print(f"PAPR Reduction: {args.papr_reduction}")
    print("\nPerformance Metrics:")
    print(f"Bit Error Rate: {results['metrics']['ber']:.2e}")
    print(f"EVM: {results['metrics']['evm']:.2f}%")
    print(f"PAPR: {10*np.log10(results['metrics']['papr']):.2f} dB")
    print(f"Processing Time: {results['metrics']['processing_time']:.4f} seconds")
    
    # Plot constellation
    plot_constellation(results['original_symbols'], args.modulation, "Transmitted Constellation")
    
    # Plot spectrum
    plot_spectrum(results['tx_signal'], title="Transmitted Signal Spectrum")
    
    # Plot channel impulse response
    plt.figure()
    plt.stem(np.abs(results['channel_taps']))
    plt.title('Channel Impulse Response')
    plt.xlabel('Tap Index')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show(block=True)


if __name__ == "__main__":
    main()