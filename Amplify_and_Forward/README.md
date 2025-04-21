# PHYSICAL Layer: Amplify And Forward Relaying Technique

This repository contains the implementation of the **Amplify-and-Forward (AF) Relaying** scheme simulation, which models a cooperative communication system. The simulation incorporates Rayleigh fading, path loss, shadowing, relay selection, maximal ratio combining (MRC), and computes key performance metrics such as Bit Error Rate (BER), Signal-to-Noise Ratio (SNR), and Spectral Efficiency.

## Overview

The **Amplify-and-Forward (AF) Relaying** scheme enhances communication reliability and coverage in wireless networks. A source node transmits data to a destination node, with the help of one or more relays. These relays amplify the signal received from the source before forwarding it to the destination. 

This simulation models the transmission over a fading channel, calculates the SNR at the destination, and computes the BER using different relaying strategies. The mathematical formulation below details each step of the simulation process.

## Mathematical Model

### 1. Transmission Power Conversion

Transmission power in dB is converted to linear scale using the following equation:

$$P_{trans} = 10^{(P_{dB} / 10)}$$

Where:
- $$P_{\text{trans}}$$: Transmission power in Watts (linear scale).
- $$P_{\text{dB}}$$: Transmission power in decibels (dB).

### 2. Channel Coefficients (Rayleigh Fading)

The channel coefficients are modeled as complex Gaussian random variables, representing Rayleigh fading:

$$
h_{sr} \sim \mathcal{CN}(0, 1), \quad h_{rd} \sim \mathcal{CN}(0, 1)
$$

- $$h_{sr} $$: Channel coefficient from Source to Relay.
- $$h_{rd} $$: Channel coefficient from Relay to Destination.

### 3. Path Loss and Shadowing

**Path loss** quantifies the reduction in signal power as it propagates through space:

$$
L = 10 \cdot \alpha \cdot \log_{10}(r)
$$

Where:
- $$\alpha $$: Path loss exponent.
- $$r $$: Distance between transmitter and receiver.

**Shadowing** adds an extra random variation to the received signal power, modeled as a Gaussian random variable:

$$
S \sim \mathcal{N}(0, \sigma_{\text{shadow}}^2)
$$

Where:
- $$\sigma_{\text{shadow}} $$: Standard deviation of the shadowing effect (dB).

### 4. Modulation

Modulated symbols are mapped to a complex constellation, e.g., for QPSK modulation:

$$
s_k = e^{j \frac{2\pi k}{M}}, \quad k \in \{0, 1, \dots, M-1\}
$$

- $$s_k $$: Modulated symbol.
- $$M $$: Modulation order (e.g., 4 for QPSK).

### 5. Source to Relay Transmission

The signal from the source to the relay is attenuated by path loss and shadowing:

$$
y_{sr} = s \cdot 10^{-\frac{L_{sr}}{10}} \cdot 10^{\frac{S_{sr}}{10}}
$$

### 6. Relay Selection and Amplification

The best relay is selected based on channel gain:

$$
i_{\text{best}} = \arg\max_{i} |h_{sr,i}|^2
$$

The relay amplifies the received signal:

$$
y_r = y_{sr,i_{\text{best}}} \cdot \sqrt{\frac{P_{\text{trans}}}{|y_{sr,i_{\text{best}}}|^2 + 1}}
$$

### 7. Relay to Destination Transmission

The relay transmits the amplified signal to the destination, affected again by path loss and shadowing:

$$
y_{rd} = y_r \cdot 10^{-\frac{L_{rd,i_{\text{best}}}}{10}} \cdot 10^{\frac{S_{rd,i_{\text{best}}}}{10}}
$$

### 8. Maximal Ratio Combining (MRC)

The signals received directly from the source and via the relay are combined to maximize signal quality:

$$
y_{\text{combined}} = y_{sr,i_{\text{best}}} + y_{rd}
$$

### 9. SNR Calculation

The SNR at the destination for the combined signal is calculated:

$$
\text{SNR}_{\text{combined}} = \frac{|y_{\text{combined}}|^2}{1 + \text{Var}(y_{\text{combined}})}
$$

### 10. Demodulation and BER Calculation

The received signal is demodulated, and the Bit Error Rate (BER) is computed:

$$
\text{BER} = \frac{\text{Number of Error Bits}}{\text{Total Number of Bits Transmitted}}
$$

### 11. Spectral Efficiency Calculation

Spectral efficiency is measured in bits per second per Hertz (bps/Hz):

$$
\eta = \log_2(M)
$$

## How to Run the Simulation
To run the simulation, simply execute the Python script. The simulation parameters (e.g., transmission power, modulation order, number of relays) can be modified within the code to experiment with different configurations. The results will include the average Bit Error Rate (BER) after the transmission of a specified number of symbols.

## Understanding the Plots
The plot shows three graphs:
	
 ### 1. BER (Bit Error Rate) vs Transmission Power (Top Graph, Red)
 
 ○ The BER initially increases and reaches a peak at around 2.5 dB, then starts to decrease with increasing transmission power. It generally follows a downward trend after 5 dB, indicating that higher transmission power reduces the BER, which is expected.
 ○ The fluctuations in the middle (around 12.5 dB) may suggest some variations in noise or relay performance that could be worth investigating further.
	
### 2. Spectral Efficiency vs Transmission Power (Middle Graph, Green)

 ○ The spectral efficiency remains constant at around 2 bits/s/Hz, which is consistent with the modulation scheme (QPSK) used in the simulation. Since the modulation order is fixed, it is expected that spectral efficiency remains unchanged despite varying transmission power.
	
 ### 3. SNR vs Transmission Power (Bottom Graph, Blue)

 ○ The SNR fluctuates significantly with transmission power. Instead of following a monotonic increase, the SNR peaks at around 2.5 dB and 10 dB, with dips in between. This suggests that either the channel model or the relay amplification process is causing irregularities in signal quality.

Summary:
	
 • The BER generally decreases with increasing transmission power, which aligns with expected behavior in communication systems, though the fluctuations indicate some irregularities in the relay or noise handling.
	
 • Spectral efficiency is constant, as expected for a fixed modulation scheme.
SNR does not increase smoothly, showing peaks and valleys across the transmission power range, which suggests there may be some optimization or modeling issues affecting the overall signal quality at various power levels.

![Plot](https://github.com/RahulSinghGulia/PHYSICAL_AmplifyAndForward/blob/main/Plots.jpg)

## Future Improvements
Support for different modulation schemes (e.g., 16-QAM, 64-QAM).
Evaluation of multiple relaying strategies (Decode-and-Forward, Selection Combining).
Simulation of multi-user or multi-hop scenarios.
