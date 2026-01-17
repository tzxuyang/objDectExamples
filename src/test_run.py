from scipy.signal import envelope
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
# (Assume 't' and 'signal' are defined as in the previous example)

duration = 1.0  # seconds
fs = 400.0  # sampling frequency
samples = int(fs * duration)
t = np.arange(samples) / fs

# Create a signal: a chirp with an amplitude modulation
signal = chirp(t, f0=20.0, t1=duration, f1=100.0, method='linear')
signal *= (1.0 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t))

# Get the upper and lower envelopes
upper_envelope, lower_envelope = envelope(signal)

plt.figure(figsize=(10, 5))
plt.plot(t, signal, label='Signal')
plt.plot(t, upper_envelope, label='Upper Envelope', color='red')
plt.plot(t, lower_envelope, label='Lower Envelope', color='red', linestyle='--')
plt.title("Signal Envelope using scipy.signal.envelope")
plt.legend()
plt.grid(True)
plt.show()