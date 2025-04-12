import numpy as np
import matplotlib.pyplot as plt

def get_modulation_config(mod_scheme):
    """
    Returns the bits per symbol and normalization factor for a given modulation scheme.
    """
    config = {
        'BPSK': {'bits_per_symbol': 1, 'norm_factor': 1.0},
        'QPSK': {'bits_per_symbol': 2, 'norm_factor': np.sqrt(2)},
        '16-QAM': {'bits_per_symbol': 4, 'norm_factor': np.sqrt(10)},
        '256-QAM': {'bits_per_symbol': 8, 'norm_factor': np.sqrt(170)},
        '1024-QAM': {'bits_per_symbol': 10, 'norm_factor': np.sqrt(682)}
    }
    return config.get(mod_scheme, None)


def symbols_gen(nsym, mod_scheme):
    """
    Generate symbols based on the selected modulation scheme.
    - BPSK: Binary Phase Shift Keying (1 bit per symbol)
    - QPSK: Quadrature Phase Shift Keying (2 bits per symbol)
    - 16-QAM: 16-QAM modulation (4 bits per symbol)
    - 256-QAM: 256-QAM modulation (8 bits per symbol)
    - 1024-QAM: 1024-QAM modulation (10 bits per symbol)

    Normalization ensures the power of the constellation points is set to a specific value (usually 1).
    This helps in controlling the average power of the transmitted signal, which is important in practical communication systems for maintaining a constant transmission power.
    """

    if mod_scheme == 'BPSK':
        # 1 bit per symbol for BPSK
        m = 1
        M = 2 ** m  # Number of symbols
        bpsk = [-1 + 0j, 1 + 0j]  # BPSK symbol values (real values)

        # Generate random integers between 0 and M-1, for the number of symbols required
        ints = np.random.randint(0, M, nsym)

        # Map the generated integers to BPSK symbols
        data = [bpsk[i] for i in ints]
        data = np.array(data, np.complex64)  # Convert to complex number array

    elif mod_scheme == 'QPSK':
        # 2 bits per symbol for QPSK
        m = 2
        M = 2 ** m  # Number of symbols
        qpsk = [1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j] / np.sqrt(2)  # QPSK symbols normalized by sqrt(2)

        # Normalize QPSK symbols by sqrt(2) to maintain unit power (average power = 1)
        # This ensures that the constellation points are normalized so the symbol energy is constant
        # Power of QPSK without normalization: 1^2 + 1^2 = 2, so normalization factor is sqrt(2)
        
        # Generate random integers for the number of QPSK symbols
        ints = np.random.randint(0, M, nsym)

        # Map the generated integers to QPSK symbols
        data = [qpsk[i] for i in ints]
        data = np.array(data, np.complex64)

    elif mod_scheme == '16-QAM':
        # 4 bits per symbol for 16-QAM
        m = 4
        M = 2 ** m  # Number of symbols

        # 16-QAM symbol values with their real and imaginary components
        qam16 = [-3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
                 -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
                  3 - 3j,  3 - 1j,  3 + 3j,  3 + 1j,
                  1 - 3j,  1 - 1j,  1 + 3j,  1 + 1j]

        # Normalize the QAM constellation to make the average symbol power = 1
        # Power of one symbol without normalization: (3^2 + 3^2) = 18
        # Average power of all symbols: (18 + 18 + 18 + 18 + ...) / 16 = 10
        # To normalize, divide each symbol by sqrt(10) to ensure the average power is 1.
        qam16 = np.array(qam16) / np.sqrt(10)

        # Generate random integers for the 16-QAM symbols
        ints = np.random.randint(0, M, nsym)

        # Map the integers to corresponding QAM symbols
        data = [qam16[i] for i in ints]
        data = np.array(data, np.complex64)

    elif mod_scheme == '256-QAM':
        # 8 bits per symbol for 256-QAM
        m = 8
        M = 2 ** m  # Number of symbols

        # Levels for the 256-QAM constellation points
        levels = np.array([-15, -13, -11, -9, -7, -5, -3, -1,
                            1,   3,   5,  7,   9, 11, 13, 15])

        # Create 256-QAM constellation by combining each level for both real and imaginary parts
        constellation = np.array([x + 1j*y for x in levels for y in levels], dtype=np.complex64)

        # Normalize the 256-QAM constellation points to have unit power
        # Power of each symbol without normalization: For example, for the point (15 + 15j), its power would be 15^2 + 15^2 = 450
        # Average power across all points: (450 + 450 + ...) / 256 = 170
        # To normalize, divide each symbol by sqrt(170) to ensure the average power per symbol is 1.
        constellation = constellation / np.sqrt(170)

        # Generate random integers for the 256-QAM symbols
        ints = np.random.randint(0, M, nsym)

        # Map the integers to corresponding 256-QAM symbols
        data = [constellation[i] for i in ints]
        data = np.array(data, np.complex64)

    elif mod_scheme == '1024-QAM':
        # 10 bits per symbol for 1024-QAM
        m = 10
        M = 2 ** m  # Number of symbols

        # Levels for the 1024-QAM constellation points
        levels = np.arange(-31, 32, 2)  # Step size of 2 for levels

        # Create 1024-QAM constellation by combining each level for both real and imaginary parts
        constellation = np.array([x + 1j*y for x in levels for y in levels], dtype=np.complex64)

        # Normalize the 1024-QAM constellation points to have unit power
        # Power of each symbol without normalization: For example, for the point (31 + 31j), its power would be 31^2 + 31^2 = 1922
        # Average power across all points: (1922 + 1922 + ...) / 1024 = 682
        # To normalize, divide each symbol by sqrt(682) to ensure the average power per symbol is 1.
        constellation = constellation / np.sqrt(682)

        # Generate random integers for the 1024-QAM symbols
        ints = np.random.randint(0, M, nsym)

        # Map the integers to corresponding 1024-QAM symbols
        data = [constellation[i] for i in ints]
        data = np.array(data, np.complex64)

    else:
        # If the modulation scheme is not one of the allowed types, raise an exception
        raise Exception('Modulation method must be BPSK, QPSK, 16-QAM, 256-QAM, or 1024-QAM.')

    return data


def scatterplot(x, y, ax=None):
    """
    Plots a 2D scatter plot representing the constellation diagram.

    Parameters:
    - x (array-like): Real part (In-phase component) of the symbols.
    - y (array-like): Imaginary part (Quadrature component) of the symbols.
    - ax (matplotlib.axes._subplots.AxesSubplot): Optional axis to plot on.

    Returns:
    - ax: The axis with the plot drawn on.
    """
    
    if ax is None:
        plt.figure(figsize=(4,4))
        ax = plt.gca()

    # Plot the constellation points
    ax.scatter(x, y, s=10, color='blue', alpha=0.7)
    
    # Add grid and axis lines
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axhline(0, color='gray', linewidth=1)
    ax.axvline(0, color='gray', linewidth=1)

    # Axis labels and title
    ax.set_title('Constellation Diagram', fontsize=14)
    ax.set_xlabel('In-phase (I)', fontsize=12)
    ax.set_ylabel('Quadrature (Q)', fontsize=12)

    # Set equal aspect ratio for better accuracy in visualization
    ax.set_aspect('equal')

    return ax


def plot_complex_signal(signal, ax=None, real_style='-b', imag_style='--r', title='I/Q Time-Domain Signal', xlabel='Time (Samples)', ylabel='Amplitude', grid=True, legend_loc='upper right', **kwargs):
    """
    Plots the real (In-phase) and imaginary (Quadrature) components of a complex signal.
    
    Parameters:
    -----------
    signal : array_like
        Complex-valued signal to plot (e.g., OFDM time-domain signal).
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    real_style : str, optional
        Line style for the real component (default: '-b' for solid blue).
    imag_style : str, optional
        Line style for the imaginary component (default: '--r' for dashed red).
    title : str, optional
        Plot title (default: 'I/Q Time-Domain Signal').
    xlabel : str, optional
        X-axis label (default: 'Time (Samples)').
    ylabel : str, optional
        Y-axis label (default: 'Amplitude').
    grid : bool, optional
        Whether to show grid lines (default: True).
    legend_loc : str, optional
        Legend location (default: 'upper right').
    **kwargs : dict
        Additional keyword arguments passed to matplotlib's plot().
    
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object with the plotted signal.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    
    # Plot real and imaginary components
    ax.plot(np.real(signal), real_style, label='In-phase (I)', **kwargs)
    ax.plot(np.imag(signal), imag_style, label='Quadrature (Q)', **kwargs)
    
    # Customize plot
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc=legend_loc, framealpha=0.8)
    ax.grid(grid)
    
    return ax


def add_awgn(signal, snr_db, measured_power=None, return_noise=False):
    """
    Adds Additive White Gaussian Noise (AWGN) to a complex signal at specified SNR.
    
    Parameters:
    -----------
    signal : array_like
        Input signal (can be real or complex).
    snr_db : float
        Desired signal-to-noise ratio in decibels (dB).
    measured_power : float, optional
        If provided, uses this as the signal power instead of calculating it.
    return_noise : bool, optional
        If True, returns both noisy signal and noise vector (default: False).
    
    Returns:
    --------
    noisy_signal : ndarray
        Input signal with AWGN added.
    noise : ndarray, optional
        Only returned if return_noise=True. The generated noise vector.
    
    Notes:
    ------
    - For complex signals, noise power is equally distributed in I & Q components.
    - SNR is defined as: SNR(dB) = 10*log10(signal_power/noise_power)
    """
    # Convert SNR from dB to linear scale
    snr_linear = 10 ** (snr_db / 10)
    
    # Calculate signal power (if not provided)
    if measured_power is None:
        signal_power = np.mean(np.abs(signal) ** 2)
    else:
        signal_power = measured_power
    
    # Calculate noise power
    noise_power = signal_power / snr_linear
    
    # Generate complex Gaussian noise
    if np.iscomplexobj(signal):
        # For complex signals: split noise power equally between I & Q
        noise_std = np.sqrt(noise_power / 2)
        noise = (noise_std * np.random.randn(len(signal)) + 
                1j * noise_std * np.random.randn(len(signal)))
    else:
        # For real signals: use full noise power in single dimension
        noise_std = np.sqrt(noise_power)
        noise = noise_std * np.random.randn(len(signal))
    
    # Add noise to signal
    noisy_signal = signal + noise
    
    if return_noise:
        return noisy_signal, noise
    return noisy_signal


def plot_constellation(received_signal, reference_symbols=None, ax=None, 
                      received_style={'s': 20, 'alpha': 0.6, 'c': 'blue'},
                      reference_style={'s': 120, 'marker': 'x', 'c': 'red'},
                      title='Signal Constellation Diagram',
                      xlabel='In-Phase Component (I)',
                      ylabel='Quadrature Component (Q)',
                      legend_loc='upper right',
                      grid=True):
    """
    Plots a constellation diagram comparing received symbols against reference points.
    
    Parameters:
    -----------
    received_signal : array_like
        Complex-valued received signal samples.
    reference_symbols : array_like, optional
        Ideal reference constellation points.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. Creates new figure if None.
    received_style : dict, optional
        Style parameters for received symbols (passed to scatter).
    reference_style : dict, optional
        Style parameters for reference symbols (passed to scatter).
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    legend_loc : str or tuple, optional
        Legend location.
    grid : bool, optional
        Whether to show grid.
    
    Returns:
    --------
    matplotlib.axes.Axes
        The axes object with the constellation plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    
    # Plot received symbols
    ax.scatter(np.real(received_signal), np.imag(received_signal), 
              label='Received Symbols', **received_style)
    
    # Plot reference constellation if provided
    if reference_symbols is not None:
        ax.scatter(np.real(reference_symbols), np.imag(reference_symbols), 
                  label='Reference Constellation', **reference_style)
    
    # Customize plot appearance
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(loc=legend_loc)
    ax.grid(grid)
    ax.axis('equal')  # Ensure proper aspect ratio
    
    return ax


def demodulate_symbols(symbols, mod_scheme):
    """Demodulates symbols to bits using the specified modulation scheme."""
    mod_config = get_modulation_config(mod_scheme)
    if not mod_config:
        raise ValueError(f"Unsupported modulation scheme: {mod_scheme}")
    
    # Get reference constellation
    ref_symbols = symbols_gen(2**mod_config['bits_per_symbol'], mod_scheme)
    unique_ref_symbols = np.unique(ref_symbols)
    
    bits = []
    for sym in symbols:
        # Find closest constellation point
        distances = np.abs(sym - unique_ref_symbols)
        closest_idx = np.argmin(distances)
        
        # Convert index to bits
        bits.extend([(closest_idx >> i) & 1 for i in range(mod_config['bits_per_symbol']-1, -1, -1)])
    
    return np.array(bits)


def plot_constellation_comparison(original, reconstructed, title='Original vs Reconstructed Symbols'):
    """
    Plots the original and reconstructed constellation symbols side-by-side.
    
    Parameters:
    - original: array-like, complex symbols (original)
    - reconstructed: array-like, complex symbols (reconstructed)
    """
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Original constellation
    scatterplot(np.real(original), np.imag(original), ax=axs[0])
    axs[0].set_title("Original Symbols")

    # Reconstructed constellation
    scatterplot(np.real(reconstructed), np.imag(reconstructed), ax=axs[1])
    axs[1].set_title("Reconstructed Symbols")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

