import numpy as np
import matplotlib.pyplot as plt

def find_delay(file1, file2):
    """
    file1 = np.loadtxt('duc_input.txt')
    file2 = np.loadtxt('ddc_output.txt')
    Loads two signal files, computes their cross-correlation, and finds
    the delay that maximizes the correlation.
    """
    try:
        # Load the input signal (reference)
        ref_signal = np.loadtxt(file1)[:, 0] # Using only the I-channel
        
        # Load the output signal (delayed)
        delayed_signal = np.loadtxt(file2)[:, 0] # Using only the I-channel

        print(f"Loaded {len(ref_signal)} samples from {file1}")
        print(f"Loaded {len(delayed_signal)} samples from {file2}")

        # Ensure signals are of the same length for correlation
        min_len = min(len(ref_signal), len(delayed_signal))
        ref_signal = ref_signal[:min_len]
        delayed_signal = delayed_signal[:min_len]

        # Normalize signals to have zero mean and unit variance
        ref_signal = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
        delayed_signal = (delayed_signal - np.mean(delayed_signal)) / np.std(delayed_signal)

        # Compute the cross-correlation
        correlation = np.correlate(delayed_signal, ref_signal, mode='full')
        
        # The lags corresponding to the correlation values
        lags = np.arange(-len(ref_signal) + 1, len(ref_signal))
        
        # Find the lag with the maximum correlation
        delay = lags[np.argmax(np.abs(correlation))]

        print("\n--- Delay Measurement Result ---")
        print(f"The measured delay is: {delay} samples.")
        print("----------------------------------")
        print(f"\nUse this value for DELAY_OFFSET in your circuit_final.cpp file.")

        # Plotting for visual verification
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.title("Signals for Correlation")
        plt.plot(ref_signal, label='DUC Input (Reference)')
        plt.plot(delayed_signal, label='DDC Output (Delayed)', alpha=0.8)
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.title(f"Cross-Correlation (Peak at delay = {delay})")
        plt.plot(lags, correlation)
        plt.axvline(delay, color='r', linestyle='--', label=f'Peak Delay: {delay}')
        plt.legend()
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.title("Aligned Signals")
        # Shift the delayed signal back by the measured delay
        if delay > 0:
            aligned_signal = delayed_signal[delay:]
            ref_plot = ref_signal[:-delay]
        else:
            aligned_signal = delayed_signal
            ref_plot = ref_signal
        plt.plot(ref_plot, label='DUC Input')
        plt.plot(aligned_signal, label='DDC Output (Aligned)', alpha=0.8, linestyle='--')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure 'duc_input.txt' and 'ddc_output.txt' exist and are not empty.")

if __name__ == "__main__":
    find_delay("duc_input.txt", "ddc_output.txt")
