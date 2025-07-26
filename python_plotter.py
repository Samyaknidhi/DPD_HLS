import numpy as np
import matplotlib.pyplot as plt

# ---- Define Functions ----
def align_signals(reference, signal_to_align):
    """
    Finds the optimal delay between two signals using cross-correlation and returns the
    aligned signal, the calculated delay, and the correlation coefficient.
    """
    # Ensure signals are 1D arrays for correlation
    ref = np.array(reference).flatten()
    sig = np.array(signal_to_align).flatten()
    
    # Truncate to the shorter length for a valid correlation
    if len(ref) != len(sig):
        print(f"Warning: Aligning signals of different lengths. Truncating to {min(len(ref), len(sig))}.")
        min_len = min(len(ref), len(sig))
        ref = ref[:min_len]
        sig = sig[:min_len]

    # Normalize signals for a clean and accurate cross-correlation to find delay
    # Add a small epsilon to std to prevent division by zero for constant signals
    ref_norm = (ref - np.mean(ref)) / (np.std(ref) + 1e-9)
    sig_norm = (sig - np.mean(sig)) / (np.std(sig) + 1e-9)

    # Perform cross-correlation
    correlation = np.correlate(ref_norm, sig_norm, mode='full')
    
    # The lag is the offset from the center of the correlation array
    delay = np.argmax(correlation) - (len(sig_norm) - 1)
    
    # Apply the shift to align the signal
    if delay > 0:  # signal_to_align is delayed relative to reference, shift it left
        aligned_signal = np.pad(sig[delay:], (0, delay), 'constant')
    elif delay < 0:  # signal_to_align is advanced, shift it right
        aligned_signal = np.pad(sig[:delay], (-delay, 0), 'constant')
    else:
        aligned_signal = sig

    # Get the correlation coefficient at the best alignment
    # Check for NaN or inf values before calculating correlation
    if np.any(np.isnan(ref)) or np.any(np.isnan(aligned_signal)) or np.any(np.isinf(ref)) or np.any(np.isinf(aligned_signal)):
        best_corr = 0
        print("Warning: NaN or Inf found in signals. Correlation cannot be computed.")
    else:
        best_corr = np.corrcoef(ref, aligned_signal)[0, 1]


    print(f"Best alignment: shift={delay}, correlation={best_corr:.4f}")
        
    return aligned_signal, delay, best_corr


def plot_spectrum(ax, signal, fs=1.0, title="Power Spectral Density", label=None):
    """Plot power spectral density of a signal with dynamic size"""
    # Get actual length of signal
    sig_len = len(signal)
    # Apply window to reduce spectral leakage
    windowed = signal * np.hamming(sig_len)
    # Calculate PSD
    psd = np.abs(np.fft.fftshift(np.fft.fft(windowed)))**2
    freq = np.fft.fftshift(np.fft.fftfreq(sig_len, d=1/fs))
    
    # Plot on log scale (dB)
    ax.plot(freq, 10*np.log10(psd + 1e-12), label=label)
    ax.set_title(title)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Power (dB)')
    ax.grid(True)
    return psd, freq

# ---- Load Data ----
try:
    dpd_i = np.loadtxt('output_dpd_i.txt')
    dpd_q = np.loadtxt('output_dpd_q.txt')
    ddc_i = np.loadtxt('output_ddc_i.txt')
    ddc_q = np.loadtxt('output_ddc_q.txt')
    pa_with_dpd = np.loadtxt('output_pa_with_dpd.txt')
    i_psf = np.loadtxt('output_psf_i.txt')
    q_psf = np.loadtxt('output_psf_q.txt')
    pa_without_dpd = np.loadtxt('output_pa.txt')
    has_psf = True
    has_without_dpd = True
    print("Successfully loaded all data files.")
except Exception as e:
    print(f"Could not load one or more data files: {e}")
    exit()

# ---- Calculate Optimal Delay for C++ code ----
# This part remains for helping tune the C++ code
_, optimal_delay_offset, _ = align_signals(dpd_i, ddc_i)
print("--------------------------------------------------")
print(f"🔬 Optimal DELAY_OFFSET Calculation for C++:")
print(f"Optimal DELAY_OFFSET should be set to: {optimal_delay_offset}")
print("--------------------------------------------------")


# ---- CORRECTED PERFORMANCE METRICS ----
print("\n--- Calculating True System Performance ---")

# The true reference is the clean signal BEFORE the DPD
reference_signal_i = i_psf
reference_signal_q = q_psf

# The signal to check is the final output from the feedback path
output_signal_i_with_dpd = ddc_i
output_signal_q_with_dpd = ddc_q

# 1. Align the final corrected output (DDC) with the original reference (PSF)
aligned_ddc_i, total_delay, corr_with_dpd = align_signals(reference_signal_i, output_signal_i_with_dpd)
aligned_ddc_q, _, _ = align_signals(reference_signal_q, output_signal_q_with_dpd)

# 2. Compensate for the system's gain
signal_power = np.mean(reference_signal_i**2 + reference_signal_q**2)
output_power = np.mean(aligned_ddc_i**2 + aligned_ddc_q**2)
gain_factor = np.sqrt(signal_power / (output_power + 1e-12))
print(f"Calculated system gain factor to be: {gain_factor:.4f}")

# Scale the output signal to match the reference power
scaled_ddc_i = aligned_ddc_i * gain_factor
scaled_ddc_q = aligned_ddc_q * gain_factor

# 3. Calculate the true NMSE
min_len_nmse = min(len(reference_signal_i), len(scaled_ddc_i))
error_power = np.mean((reference_signal_i[:min_len_nmse] - scaled_ddc_i[:min_len_nmse])**2 + 
                      (reference_signal_q[:min_len_nmse] - scaled_ddc_q[:min_len_nmse])**2)

nmse_total = 10 * np.log10(error_power / signal_power)
print(f"✅ Final Corrected NMSE: {nmse_total:.2f} dB")


# ---- Plotting ----
fig = plt.figure(figsize=(18, 24))
grid = plt.GridSpec(6, 2, figure=fig, hspace=0.5, wspace=0.3)
fig.suptitle(f'DPD/PA System Analysis (WITH DPD)\nCorrected NMSE: {nmse_total:.2f} dB', fontsize=16)

# 1. DPD Error (PSF vs DPD Output)
ax1 = fig.add_subplot(grid[0, 0])
err_i = i_psf - dpd_i
err_q = q_psf - dpd_q
ax1.plot(err_i, label='Error I')
ax1.plot(err_q, label='Error Q')
ax1.set_title('DPD Error Signal (PSF vs DPD Out)')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Error')
ax1.legend()
ax1.grid(True)

# 2. AM-AM (Reference vs Corrected Output)
ax2 = fig.add_subplot(grid[0, 1])
amp_in_ref = np.sqrt(i_psf**2 + q_psf**2)
amp_out_corr = np.sqrt(scaled_ddc_i**2 + scaled_ddc_q**2)
ax2.scatter(amp_in_ref, amp_out_corr, s=2, alpha=0.5)
ax2.plot([0, np.max(amp_in_ref)], [0, np.max(amp_in_ref)], 'r--', label='Ideal')
ax2.set_title('AM-AM Characteristic')
ax2.set_xlabel('Reference Magnitude')
ax2.set_ylabel('Corrected Output Magnitude')
ax2.grid(True)
ax2.legend()

# 3. AM-PM
ax3 = fig.add_subplot(grid[1, 0])
phase_in = np.unwrap(np.arctan2(q_psf, i_psf))
phase_out = np.unwrap(np.arctan2(scaled_ddc_q, scaled_ddc_i))
phase_diff = phase_out - phase_in
ax3.scatter(amp_in_ref, phase_diff, s=2, alpha=0.5)
ax3.set_title('AM-PM Characteristic')
ax3.set_xlabel('Reference Magnitude')
ax3.set_ylabel('Phase Error (rad)')
ax3.grid(True)

# 4. DDC Output Constellation
ax4 = fig.add_subplot(grid[1, 1])
ax4.scatter(scaled_ddc_i, scaled_ddc_q, s=2, alpha=0.5)
ax4.set_title('Corrected Output Constellation')
ax4.set_xlabel('I')
ax4.set_ylabel('Q')
ax4.axis('equal')
ax4.grid(True)

# 5. Histogram of Output Magnitude
ax5 = fig.add_subplot(grid[2, 0])
ax5.hist(amp_out_corr, bins=100, alpha=0.7, label='Corrected DDC Output')
ax5.set_title('Histogram of Corrected Output Magnitude')
ax5.set_xlabel('Magnitude')
ax5.set_ylabel('Count')
ax5.legend()
ax5.grid(True)

# 6. Time-domain overlay: All signals (I channel)
n_show = 200
ax6 = fig.add_subplot(grid[2, 1])
ax6.plot(i_psf[:n_show], label='Reference I')
ax6.plot(scaled_ddc_i[:n_show], label='Corrected DDC I', alpha=0.7, linestyle='--')
ax6.set_title('Time-Domain: Signal Comparison')
ax6.set_xlabel('Sample')
ax6.set_ylabel('Amplitude')
ax6.legend()
ax6.grid(True)

# 7. Normalized Input vs Output comparison (shape matching)
ax7 = fig.add_subplot(grid[3, 0])
norm_input_i = i_psf[:n_show] / (np.max(np.abs(i_psf[:n_show])) + 1e-9)
norm_output_i = scaled_ddc_i[:n_show] / (np.max(np.abs(scaled_ddc_i[:n_show])) + 1e-9)
ax7.plot(norm_input_i, label='Normalized Reference')
ax7.plot(norm_output_i, label='Normalized Corrected DDC', alpha=0.7, linestyle='--')
ax7.set_title('Shape Comparison (Normalized)')
ax7.set_xlabel('Sample')
ax7.set_ylabel('Normalized Amplitude')
ax7.legend()
ax7.grid(True)

# 8. Correlation scatter plot: Reference vs Corrected Output
ax8 = fig.add_subplot(grid[3, 1])
ax8.scatter(i_psf, scaled_ddc_i, s=1, alpha=0.5)
ax8.set_title(f'Reference vs. Corrected Output Correlation (r={corr_with_dpd:.4f})')
ax8.set_xlabel('Reference I')
ax8.set_ylabel('Corrected DDC I (Aligned & Scaled)')
lims = [np.min([ax8.get_xlim(), ax8.get_ylim()]), np.max([ax8.get_xlim(), ax8.get_ylim()])]
ax8.plot(lims, lims, 'r-', alpha=0.75, label='Ideal (y=x)')
ax8.legend()
ax8.grid(True)

# 9. PA Output with DPD - Magnitude Analysis
ax9 = fig.add_subplot(grid[4, 0])
ax9.plot(np.abs(pa_with_dpd[:n_show*8]), color='red') # Show more samples for wideband signal
ax9.set_title('PA Output w/DPD Magnitude')
ax9.set_xlabel('Sample (at PA rate)')
ax9.set_ylabel('Amplitude')
ax9.grid(True)

# 10. DPD Input vs PA Output (Direct Comparison with Scaling)
ax10 = fig.add_subplot(grid[4, 1])
# Upsample PSF signal to match PA rate for comparison
psf_up = np.repeat(i_psf, 8)
pa_with_dpd_aligned, _, _ = align_signals(psf_up, pa_with_dpd)
scale_factor = np.mean(np.abs(pa_with_dpd_aligned)) / (np.mean(np.abs(psf_up)) + 1e-9)
scaled_input = psf_up * scale_factor
ax10.plot(scaled_input[:n_show*8], label=f'Scaled Reference (×{scale_factor:.2f})')
ax10.plot(pa_with_dpd_aligned[:n_show*8], color='red', label='PA Output w/DPD', alpha=0.7)
ax10.set_title('Reference vs PA Output')
ax10.set_xlabel('Sample (at PA rate)')
ax10.set_ylabel('Amplitude')
ax10.legend()
ax10.grid(True)

# 11. Spectral Analysis - Reference Signal
ax11 = fig.add_subplot(grid[5, 0])
fft_len = min(4096, len(i_psf))
plot_spectrum(ax11, i_psf[:fft_len], title="Reference Signal Spectrum (PSF)")
ax11.set_xlim([-0.5, 0.5])

# 12. Spectral Analysis - PA Output with DPD
ax12 = fig.add_subplot(grid[5, 1])
fft_len_pa = min(4096, len(pa_with_dpd))
plot_spectrum(ax12, pa_with_dpd[:fft_len_pa], fs=8.0, title="PA Output w/DPD Spectrum")
ax12.set_xlim([-4.0, 4.0])

# Add performance metrics text box
metrics_text = f"Performance Metrics:\n" \
               f"Corrected NMSE: {nmse_total:.2f} dB\n" \
               f"System Delay: {total_delay} samples\n" \
               f"System Gain Factor: {gain_factor:.4f}"
fig.text(0.02, 0.01, metrics_text, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# ---- Create Without DPD Figure ----
if has_without_dpd:
    fig2 = plt.figure(figsize=(18, 20))
    grid2 = plt.GridSpec(5, 2, figure=fig2, hspace=0.5, wspace=0.3)
    
    # Align the uncorrected PA output (decimated) with the clean reference
    aligned_pa_no_dpd, shift_no_dpd, corr_no_dpd = align_signals(i_psf, pa_without_dpd[::8])
    
    fig2.suptitle(f'System Analysis WITHOUT DPD\nAlignment: {shift_no_dpd}, Correlation: {corr_no_dpd:.4f}',
                  fontsize=16)

    # 1. AM-AM Without DPD
    ax2_1 = fig2.add_subplot(grid2[0, 0])
    ax2_1.scatter(amp_in_ref, np.abs(aligned_pa_no_dpd), s=2, alpha=0.5, color='green')
    ax2_1.plot([0, np.max(amp_in_ref)], [0, np.max(amp_in_ref)], 'r--', label='Ideal')
    ax2_1.set_title('AM-AM Characteristic (Without DPD)')
    ax2_1.set_xlabel('Reference Magnitude')
    ax2_1.set_ylabel('Distorted Output Magnitude')
    ax2_1.grid(True)
    ax2_1.legend()
    
    # 2. Time-domain comparison
    ax2_2 = fig2.add_subplot(grid2[0, 1])
    ax2_2.plot(i_psf[:n_show], label='Reference I')
    ax2_2.plot(aligned_pa_no_dpd[:n_show], label='PA Output (No DPD)', color='green', linestyle='--')
    ax2_2.set_title('Time-Domain: Signal Comparison')
    ax2_2.set_xlabel('Sample')
    ax2_2.set_ylabel('Amplitude')
    ax2_2.legend()
    ax2_2.grid(True)
    
    # 3. Normalized comparison (shape matching)
    ax2_3 = fig2.add_subplot(grid2[1, 0])
    norm_input_i = i_psf[:n_show] / (np.max(np.abs(i_psf[:n_show])) + 1e-9)
    norm_pa_no_dpd = aligned_pa_no_dpd[:n_show] / (np.max(np.abs(aligned_pa_no_dpd[:n_show])) + 1e-9)
    ax2_3.plot(norm_input_i, label='Normalized Reference')
    ax2_3.plot(norm_pa_no_dpd, label='Normalized PA (No DPD)', color='green')
    ax2_3.set_title('Shape Comparison (Normalized)')
    ax2_3.set_xlabel('Sample')
    ax2_3.set_ylabel('Normalized Amplitude')
    ax2_3.legend()
    ax2_3.grid(True)
    
    # 4. Correlation scatter plot
    ax2_4 = fig2.add_subplot(grid2[1, 1])
    ax2_4.scatter(i_psf, aligned_pa_no_dpd, s=1, alpha=0.5, color='green')
    ax2_4.set_title(f'Reference vs PA Output Correlation (r={corr_no_dpd:.4f})')
    ax2_4.set_xlabel('Reference I')
    ax2_4.set_ylabel('PA Output (No DPD)')
    lims = [np.min([ax2_4.get_xlim(), ax2_4.get_ylim()]),  
            np.max([ax2_4.get_xlim(), ax2_4.get_ylim()])]
    ax2_4.plot(lims, lims, 'r-', alpha=0.5, label='Ideal (y=x)')
    ax2_4.legend()
    ax2_4.grid(True)
    
    # 5. Spectral Regrowth Comparison
    ax2_6 = fig2.add_subplot(grid2[2, :])
    pa_with_dpd_aligned_spec, _, _ = align_signals(pa_without_dpd, pa_with_dpd)
    plot_spectrum(ax2_6, pa_without_dpd, fs=8.0, label='Without DPD')
    plot_spectrum(ax2_6, pa_with_dpd_aligned_spec, fs=8.0, label='With DPD')
    ax2_6.set_title('Spectral Regrowth Comparison')
    ax2_6.legend()
    
    # 6. ACPR Calculation
    fft_len = min(4096, len(pa_with_dpd_aligned_spec), len(pa_without_dpd))
    freq = np.fft.fftshift(np.fft.fftfreq(fft_len, d=1/8.0))
    with_dpd_psd = np.abs(np.fft.fftshift(np.fft.fft(pa_with_dpd_aligned_spec[:fft_len] * np.hamming(fft_len))))**2
    no_dpd_psd = np.abs(np.fft.fftshift(np.fft.fft(pa_without_dpd[:fft_len] * np.hamming(fft_len))))**2
    
    in_band = (np.abs(freq) < 0.5)
    adjacent_band = (np.abs(freq) > 0.75) & (np.abs(freq) < 1.25)
    
    acpr_with_dpd = 10 * np.log10(np.sum(with_dpd_psd[adjacent_band]) / (np.sum(with_dpd_psd[in_band]) + 1e-12))
    acpr_no_dpd = 10 * np.log10(np.sum(no_dpd_psd[adjacent_band]) / (np.sum(no_dpd_psd[in_band]) + 1e-12))

    # 7. ACPR Bar Chart
    ax2_7 = fig2.add_subplot(grid2[3, 0])
    bar_labels = ['Without DPD', 'With DPD']
    acpr_values = [acpr_no_dpd, acpr_with_dpd]
    ax2_7.bar(bar_labels, acpr_values, color=['green', 'blue'])
    ax2_7.set_title('Adjacent Channel Power Ratio (ACPR)')
    ax2_7.set_ylabel('ACPR (dBc)')
    ax2_7.grid(axis='y')
    for i, v in enumerate(acpr_values):
        ax2_7.text(i, v - 5, f"{v:.2f}", ha='center', color='white', fontweight='bold')

    # 8. NMSE Comparison
    ax2_8 = fig2.add_subplot(grid2[3, 1])
    # CORRECTED NMSE for "Without DPD" case
    scaled_pa_no_dpd_i, _, _ = align_signals(i_psf, pa_without_dpd[::8])
    scaled_pa_no_dpd_q, _, _ = align_signals(q_psf, np.zeros_like(pa_without_dpd[::8])) # Assume Q is 0 for real PA output
    gain_factor_no_dpd = np.sqrt(signal_power / (np.mean(scaled_pa_no_dpd_i**2 + scaled_pa_no_dpd_q**2) + 1e-12))
    scaled_pa_no_dpd_i *= gain_factor_no_dpd
    scaled_pa_no_dpd_q *= gain_factor_no_dpd
    
    min_len_no_dpd = min(len(i_psf), len(scaled_pa_no_dpd_i))
    error_power_no_dpd = np.mean((i_psf[:min_len_no_dpd] - scaled_pa_no_dpd_i[:min_len_no_dpd])**2 + 
                                 (q_psf[:min_len_no_dpd] - scaled_pa_no_dpd_q[:min_len_no_dpd])**2)
    nmse_no_dpd = 10 * np.log10(error_power_no_dpd / signal_power)
    
    bar_labels = ['Without DPD', 'With DPD']
    nmse_values = [nmse_no_dpd, nmse_total]
    ax2_8.bar(bar_labels, nmse_values, color=['green', 'blue'])
    ax2_8.set_title('NMSE Comparison')
    ax2_8.set_ylabel('NMSE (dB)')
    ax2_8.grid(axis='y')
    for i, v in enumerate(nmse_values):
        ax2_8.text(i, v - 2, f"{v:.2f}", ha='center', color='white', fontweight='bold')

    # 9. Time-domain comparison with scaled inputs for both
    ax2_9 = fig2.add_subplot(grid2[4, 0])
    scale_factor_no_dpd = np.mean(np.abs(aligned_pa_no_dpd[:n_show])) / (np.mean(np.abs(i_psf[:n_show])) + 1e-9)
    scaled_input_no_dpd = i_psf[:n_show] * scale_factor_no_dpd
    
    ax2_9.plot(scaled_input_no_dpd, label=f'Scaled Reference (×{scale_factor_no_dpd:.2f})')
    ax2_9.plot(aligned_pa_no_dpd[:n_show], label='PA Output (No DPD)', color='green', linestyle='--')
    ax2_9.set_title(f'Reference vs PA Output Without DPD (NMSE: {nmse_no_dpd:.2f} dB)')
    ax2_9.set_xlabel('Sample')
    ax2_9.set_ylabel('Amplitude')
    ax2_9.legend()
    ax2_9.grid(True)
    
    # 10. Spectral mask comparison
    ax2_10 = fig2.add_subplot(grid2[4, 1])
    # Create a simple spectral mask
    mask = np.ones_like(freq) * -50
    mask[(freq > -0.15) & (freq < 0.15)] = -20  # In-band region
    
    ax2_10.plot(freq, 10*np.log10(with_dpd_psd + 1e-12), label='With DPD')
    ax2_10.plot(freq, 10*np.log10(no_dpd_psd + 1e-12), label='Without DPD') 
    ax2_10.plot(freq, mask, 'k--', label='Spectral Mask')
    ax2_10.set_title('Spectral Mask Compliance')
    ax2_10.set_xlabel('Normalized Frequency')
    ax2_10.set_ylabel('Power (dB)')
    ax2_10.legend()
    ax2_10.set_xlim([-0.5, 0.5])
    ax2_10.grid(True)
    
    # Add performance metrics text box
    metrics_text = f"Without DPD Performance:\n" \
                   f"NMSE: {nmse_no_dpd:.2f} dB\n" \
                   f"ACPR: {acpr_no_dpd:.2f} dB\n" \
                   f"Spectral Regrowth Improvement: {acpr_no_dpd - acpr_with_dpd:.2f} dB"
    fig2.text(0.5, 0.01, metrics_text, fontsize=12, ha='center',
              bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

# Show all figures
plt.show()
