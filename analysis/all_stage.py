import os
import numpy as np
import matplotlib.pyplot as plt

def plot_time_domain(data, title, filename):
    """Plots the time-domain signal."""
    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title(title)
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_psd(data, title, filename):
    """Plots the Power Spectral Density (PSD) of the signal."""
    plt.figure(figsize=(10, 6))
    plt.psd(data, NFFT=1024, Fs=1.0, scale_by_freq=True)
    plt.title(title)
    plt.xlabel("Normalized Frequency")
    plt.ylabel("Power/Frequency (dB/Hz)")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_constellation(i_data, q_data, title, filename):
    """Plots the constellation diagram."""
    plt.figure(figsize=(8, 8))
    plt.scatter(i_data, q_data, s=1) # s is the marker size
    plt.title(title)
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(filename)
    plt.close()

def main():
    # List of your output files
    files = [
        "output_adc_i.txt",
        "output_conste_i.txt",
        "output_dac_i.txt",
        "output_ddc_i.txt",
        "output_dpd_i.txt",
        "output_duc.txt",
        "output_pa_with_dpd.txt",
        "output_psf_fb_i.txt",
        "output_psf_i.txt",
        "output_qm.txt",
    ]

    # Create a directory to save the plots
    output_dir = "plots"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in files:
        if not os.path.exists(file):
            print(f"File not found: {file}. Skipping.")
            continue

        print(f"Processing {file}...")
        try:
            # For constellation and QM files, assume two columns (I and Q)
            if "conste" in file or "qm" in file:
                data = np.loadtxt(file)
                if data.ndim == 2 and data.shape[1] >= 2:
                    i_data = data[:, 0]
                    q_data = data[:, 1]
                    title = f"Constellation Diagram - {file}"
                    filename = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_constellation.png")
                    plot_constellation(i_data, q_data, title, filename)
                else:
                     print(f"Warning: {file} does not have two columns. Skipping constellation plot.")
            # For all other files, assume a single column
            else:
                data = np.loadtxt(file)

                # Time-domain plot
                title = f"Time Domain Signal - {file}"
                filename = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_time.png")
                plot_time_domain(data, title, filename)

                # PSD plot
                title = f"Power Spectral Density - {file}"
                filename = os.path.join(output_dir, f"{os.path.splitext(file)[0]}_psd.png")
                plot_psd(data, title, filename)
        except Exception as e:
            print(f"Error processing {file}: {e}")

    print("\nAll plots have been generated and saved in the 'plots' directory.")

if __name__ == "__main__":
    main()