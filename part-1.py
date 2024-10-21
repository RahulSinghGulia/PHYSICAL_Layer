import numpy as np
import matplotlib.pyplot as plt

# Set the parameters for frequency hopping
frequencies = [2.4, 2.45, 2.5]  # GHz, example frequency hopping
hopping_pattern = np.random.choice(frequencies, size=100)  # Random hopping pattern over 100 time steps

# Example plaintext message (binary data) to transmit
plaintext = np.random.randint(0, 2, 100)  # 100 bits of random binary data

# Define XOR-based encryption
def xor_encrypt_decrypt(data, key):
    return np.bitwise_xor(data, key)

# Generate a random encryption key
key = np.random.randint(0, 2, 100)  # 100 bits encryption key

# Encrypt the message using XOR encryption
encrypted_message = xor_encrypt_decrypt(plaintext, key)

# Simulate jamming as random noise on certain frequencies
def simulate_jamming(hopping_pattern, frequencies, jamming_frequency):
    jammed_signal = np.zeros_like(hopping_pattern)
    for i, freq in enumerate(hopping_pattern):
        if freq == jamming_frequency:
            jammed_signal[i] = 1  # Jamming occurs at this frequency
    return jammed_signal

# Jammer jams at 2.45 GHz
jamming_frequency = 2.45
jammed_signal = simulate_jamming(hopping_pattern, frequencies, jamming_frequency)

# Simulate frequency-hopping modulation for transmission
def frequency_hopping_transmit(data, hopping_pattern, jammed_signal):
    transmitted_signal = []
    for i in range(len(data)):
        if jammed_signal[i] == 1:
            transmitted_signal.append(0)  # Jammed, no signal transmitted
        else:
            transmitted_signal.append(data[i])  # Transmit encrypted data
    return np.array(transmitted_signal)

# Transmit encrypted message with frequency hopping
transmitted_signal = frequency_hopping_transmit(encrypted_message, hopping_pattern, jammed_signal)

# Receiver-side decryption
def frequency_hopping_receive(transmitted_signal, hopping_pattern, jammed_signal):
    received_signal = []
    for i in range(len(transmitted_signal)):
        if jammed_signal[i] == 1:
            received_signal.append(0)  # Jammed, lost signal
        else:
            received_signal.append(transmitted_signal[i])  # Received signal
    return np.array(received_signal)
received_signal = frequency_hopping_receive(transmitted_signal, hopping_pattern, jammed_signal)

# Decrypt the received message
decrypted_message = xor_encrypt_decrypt(received_signal, key)

# Compare original and decrypted message
errors = np.sum(plaintext != decrypted_message)
print(f"Number of errors in decrypted message: {errors}")

# Plot the results
plt.figure(figsize=(12, 6))

# Plot frequency hopping pattern
plt.subplot(3, 1, 1)
plt.plot(hopping_pattern, 'o-', label='Frequency Hopping Pattern (GHz)')
plt.axhline(y=jamming_frequency, color='r', linestyle='--', label='Jamming Frequency')
plt.title('Frequency Hopping and Jamming')
plt.xlabel('Time Step')
plt.ylabel('Frequency (GHz)')
plt.legend()

# Plot the transmitted signal
plt.subplot(3, 1, 2)
plt.step(range(len(transmitted_signal)), transmitted_signal, where='mid', label='Transmitted Signal (Encrypted)')
plt.title('Transmitted Signal with Encryption and Jamming Resistance')
plt.xlabel('Time Step')
plt.ylabel('Signal')

# Plot the decrypted signal
plt.subplot(3, 1, 3)
plt.step(range(len(decrypted_message)), decrypted_message, where='mid', label='Decrypted Message', color='green')
plt.title('Decrypted Message After Receiving')
plt.xlabel('Time Step')
plt.ylabel('Decrypted Signal')
plt.tight_layout()
plt.show()

# Final message summary
print("Original Message:    ", plaintext)
print("Decrypted Message:   ", decrypted_message)
print("Errors:              ", errors)
