import adi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import time

# Pluto SDR Configuration
sdr_ip = "ip:192.168.2.1"
my_sdr = adi.ad9361(uri=sdr_ip)

sample_rate = 0.6e6  # 600 kHz
center_freq = 2.3e9  # 2.3 GHz
signal_freq = 100e3  # 100 kHz
fft_size = 1024 * 4  # FFT size
c = 3e8  # Speed of light (m/s)

# Configure SDR Rx
my_sdr.sample_rate = int(sample_rate)
my_sdr.rx_lo = int(center_freq)
my_sdr.rx_enabled_channels = [0, 1]
my_sdr.rx_buffer_size = int(fft_size)
my_sdr.gain_control_mode_chan0 = "manual"
my_sdr.gain_control_mode_chan1 = "manual"
my_sdr.rx_hardwaregain_chan0 = int(50)
my_sdr.rx_hardwaregain_chan1 = int(50)
my_sdr.rx_quadrature_tracking_en = True
my_sdr.rx_phase_correction_en = True

# Configure SDR Tx
my_sdr.tx_lo = int(center_freq)
my_sdr.tx_enabled_channels = [0, 1]
my_sdr.tx_cyclic_buffer = True
my_sdr.tx_hardwaregain_chan0 = -5
my_sdr.tx_hardwaregain_chan1 = -5
my_sdr.tx_quadrature_tracking_en = True

# Generate and Transmit a Continuous Wave (CW) Signal
t = np.arange(0, 1, 1 / sample_rate)  # Time vector
tx_signal = 0.5 * np.exp(2j * np.pi * signal_freq * t)  # Complex sinusoid

my_sdr.tx([tx_signal, tx_signal])  # Transmit the signal

# Doppler Shift Estimation Function
def estimate_doppler(rx_data, sample_rate, center_freq):
    """Estimate Doppler shift from received signal."""
    f, Pxx = welch(rx_data, fs=sample_rate, nperseg=fft_size)
    peak_freq = f[np.argmax(Pxx)]  # Find peak frequency
    doppler_shift = peak_freq # Compute shift
    velocity = (doppler_shift * c) / (2 * center_freq)  # Doppler formula
    return doppler_shift, velocity

# Data Collection
duration = 10  # Duration to collect data (seconds)
velocity_data = []
timestamps = []

start_time = time.time()
while time.time() - start_time < duration:
    rx_samples = my_sdr.rx()  # Receive samples
    rx_signal = rx_samples[0] + rx_samples[1]  # Combine I/Q data
    doppler_shift, velocity = estimate_doppler(rx_signal, sample_rate, center_freq)
    velocity_data.append(velocity)
    timestamps.append(time.time() - start_time)
    print(f"Time: {timestamps[-1]:.2f}s | Doppler Shift: {doppler_shift:.2f} Hz | Velocity: {velocity:.2f} m/s")

# Save Data
np.savetxt("doppler_velocity_data.csv", np.column_stack((timestamps, velocity_data)), delimiter=",", header="Time (s), Velocity (m/s)")

# Plot Velocity vs Time
plt.figure(figsize=(10, 5))
plt.plot(timestamps, velocity_data, label="Relative Velocity (m/s)")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Relative Velocity Over Time")
plt.legend()
plt.grid()
plt.show()
