from Crypto.Cipher import AES
from secrets import token_bytes
import random
import time
import numpy as np
import matplotlib.pyplot as plt

# Key generation 
key = token_bytes(16)

def encrypt(msg):
    cipher = AES.new(key, AES.MODE_EAX)
    nonce = cipher.nonce    
    ciphertext, tag = cipher.encrypt_and_digest(msg.encode('ascii'))
    return nonce, ciphertext, tag

def decrypt(nonce, ciphertext, tag):
    cipher = AES.new(key, AES.MODE_EAX, nonce=nonce)
    plaintext = cipher.decrypt(ciphertext)
    try:
        cipher.verify(tag)
        return plaintext.decode('ascii')
    except ValueError:
        return False

def awgn_channel(signal, snr_dB):
    """ Simulate an AWGN channel by adding noise to the signal. """
    snr = 10 ** (snr_dB / 10)  # Convert dB to a linear scale
    signal_power = np.mean(np.abs(signal)**2)  # Calculate signal power
    noise_power = signal_power / snr  # Calculate noise power
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)  # Generate Gaussian noise
    return signal + noise.astype(np.uint8)  # Add noise to the signal and return

def transmit(nonce, ciphertext, tag, error_rate=0, jam_steps=[], snr_dB=30):
    """ Simulate transmission, with a chance of introducing errors and jamming. """
    current_time = int(time.time() * 100)  # Get current time in milliseconds
    frequency = get_frequency(current_time)

    if current_time % 100 in jam_steps:  # Check if the current time step is jammed
        return nonce, b'\x00' * len(ciphertext), tag, 0  # Transmit zeroed ciphertext if jammed
    
    # Add noise to the ciphertext using AWGN channel model
    noisy_ciphertext = awgn_channel(np.frombuffer(ciphertext, dtype=np.uint8), snr_dB)
    
    # Track errors introduced during transmission
    errors = 0
    if random.random() < error_rate:
        # Introduce a simple error (e.g., corrupt the noisy ciphertext)
        corrupted_ciphertext = bytearray(noisy_ciphertext)
        corrupted_byte_index = random.randint(0, len(corrupted_ciphertext) - 1)
        corrupted_ciphertext[corrupted_byte_index] ^= 0xFF  # Flip a random byte
        errors += 1  # Count the error
        return nonce, bytes(corrupted_ciphertext), tag, errors
    
    return nonce, bytes(noisy_ciphertext), tag, errors

def get_frequency(current_time):
    """ Simulate frequency hopping based on time. """
    frequencies = [2.4, 2.45, 2.5, 2.55]  # List of possible frequencies (in GHz)
    return frequencies[(current_time // 100) % len(frequencies)]  # Select frequency based on time step

def receive(nonce, ciphertext, tag):
    """ Simulate receiving the message and attempting decryption. """
    return decrypt(nonce, ciphertext, tag)

# Directly input the message string
msg = "Encrypting messages using AES"

# Performance metrics collection
success_rates = []
error_rates = np.linspace(0, 0.5, 10)  # Varying error rates from 0% to 50%
snr_values = np.arange(0, 40, 5)  # Varying SNR from 0 to 100 dB
jam_steps = [0, 1, 2, 5, 6, 8, 10]  # Specify the time steps when jamming occurs
num_trials = 100  # Number of trials for each error rate

# Evaluate transmission success rates across different error rates
for error_rate in error_rates:
    successful_transmissions = 0
    total_errors = 0
    for _ in range(num_trials):
        if msg:
            nonce, ciphertext, tag = encrypt(msg)
            nonce_transmitted, ciphertext_transmitted, tag_transmitted, errors = transmit(nonce, ciphertext, tag, error_rate, jam_steps, snr_dB=30)
            plaintext = receive(nonce_transmitted, ciphertext_transmitted, tag_transmitted)
            total_errors += errors
            if plaintext:  # Count successful decryptions
                successful_transmissions += 1
    success_rates.append(successful_transmissions / num_trials)

# Plot Error Rate vs. Transmission Success Rate
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(error_rates, success_rates, marker='o')
plt.title('Error Rate vs. Transmission Success Rate')
plt.xlabel('Error Rate')
plt.ylabel('Success Rate')
plt.grid()

# Evaluate transmission success rates across different SNR values
success_rates_snr = []
for snr in snr_values:
    successful_transmissions = 0
    total_errors = 0
    for _ in range(num_trials):
        if msg:
            nonce, ciphertext, tag = encrypt(msg)
            nonce_transmitted, ciphertext_transmitted, tag_transmitted, errors = transmit(nonce, ciphertext, tag, error_rate=0.1, jam_steps=jam_steps, snr_dB=snr)
            plaintext = receive(nonce_transmitted, ciphertext_transmitted, tag_transmitted)
            total_errors += errors
            if plaintext:  # Count successful decryptions
                successful_transmissions += 1
    success_rates_snr.append(successful_transmissions / num_trials)

# Plot SNR vs. Transmission Success Rate
plt.subplot(1, 2, 2)
plt.plot(snr_values, success_rates_snr, marker='o', color='orange')
plt.title('SNR vs. Transmission Success Rate')
plt.xlabel('SNR (dB)')
plt.ylabel('Success Rate')
plt.grid()

plt.tight_layout()
plt.show()

# Print messages and errors
print(f'Plaintext: {msg}')
print(f'Encrypted ciphertext (hex): {ciphertext.hex()}')
print(f'Decrypted Plaintext (after last trial): {plaintext}')
print(f'Total errors encountered during the last transmission: {total_errors}')
