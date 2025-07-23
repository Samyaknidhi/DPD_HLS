import numpy as np
import matplotlib.pyplot as plt

# ---- Define Functions ----
def align_signals(reference, signal_to_align, max_shift=100):
    best_corr = 0
    best_shift = 0
    
    for shift in range(-max_shift, max_shift):
        if shift > 0:
            corr = np.corrcoef(reference[shift:], signal_to_align[:-shift])[0,1]
        elif shift < 0:
            corr = np.corrcoef(reference[:shift], signal_to_align[-shift:])[0,1]
        else:
            corr = np.corrcoef(reference, signal_to_align)[0,1]
            
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_shift = shift
    
    print(f"Best alignment: shift={best_shift}, correlation={best_corr:.4f}")
    
    if best_shift > 0:
        aligned = np.zeros_like(signal_to_align)
        aligned[best_shift:] = signal_to_align[:-best_shift]
    elif best_shift < 0:
        aligned = np.zeros_like(signal_to_align)
        aligned[:best_shift] = signal_to_align[-best_shift:]
    else:
        aligned = signal_to_align
        
    return aligned, best_shift, best_corr

# ---- Load Data ----
dpd_i = np.loadtxt('output_dpd_i.txt')
dpd_q = np.loadtxt('output_dpd_q.txt')

# Load DDC output (after PA and DDC)
ddc_i = np.loadtxt('output_ddc_i.txt')
ddc_q = np.loadtxt('output_ddc_q.txt')

# Load PA output with DPD
pa_with_dpd = np.loadtxt('output_pa_with_dpd.txt')

# Optionally, reference PSF (if available)
try:
    i_psf = np.loadtxt('output_psf_i.txt')
    q_psf = np.loadtxt('output_psf_q.txt')
    has_psf = True
except Exception:
    has_psf = False

# ---- Align Lengths ----
interp = len(ddc_i) // len(dpd_i)
dpd_i_up = np.repeat(dpd_i, interp)
dpd_q_up = np.repeat(dpd_q, interp)

# Determine number of points to show
n_show = min(1000, len(ddc_i), len(pa_with_dpd))  

# Align PA output signal with input
pa_with_dpd_aligned, shift, corr = align_signals(dpd_i_up[:n_show], pa_with_dpd[:n_show])

# ---- Compute Features ----
amp_in = np.sqrt(dpd_i_up**2 + dpd_q_up**2)
amp_out = np.sqrt(ddc_i**2 + ddc_q**2)
in_phase = np.arctan2(dpd_q_up, dpd_i_up)
out_phase = np.arctan2(ddc_q, ddc_i)
phase_diff = np.unwrap(out_phase) - np.unwrap(in_phase)

# ---- Calculate Performance Metrics ----
# NMSE calculation (Normalized Mean Square Error)
nmse_i = 10 * np.log10(np.mean((dpd_i_up - ddc_i)**2) / np.mean(dpd_i_up**2))
nmse_q = 10 * np.log10(np.mean((dpd_q_up - ddc_q)**2) / np.mean(dpd_q_up**2))
nmse_total = 10 * np.log10(np.mean((dpd_i_up - ddc_i)**2 + (dpd_q_up - ddc_q)**2) / 
                           np.mean(dpd_i_up**2 + dpd_q_up**2))

# Calculate NMSE between DPD input and aligned PA output
nmse_pa_dpd = 10 * np.log10(np.mean((dpd_i_up[:n_show] - pa_with_dpd_aligned)**2) / 
                            np.mean(dpd_i_up[:n_show]**2))

# ---- Plotting ----
fig = plt.figure(figsize=(18, 20))  # Increased size for 5 rows
grid = plt.GridSpec(5, 2, figure=fig)  # Added one more row

fig.suptitle(f'DPD/PA System Analysis (After DDC)\nNMSE: {nmse_total:.2f} dB', fontsize=16)

# 1. DPD Error
ax1 = fig.add_subplot(grid[0, 0])
if has_psf:
    err_i = i_psf - dpd_i
    err_q = q_psf - dpd_q
    ax1.plot(err_i, label='Error I')
    ax1.plot(err_q, label='Error Q')
    ax1.set_title('DPD Error Signal')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Error')
    ax1.legend()
    ax1.grid(True)
else:
    ax1.axis('off')
    ax1.text(0.5, 0.5, 'No PSF reference\n(error plot skipped)', 
             ha='center', va='center', fontsize=12)

# 2. AM-AM
ax2 = fig.add_subplot(grid[0, 1])
ax2.scatter(amp_in, amp_out, s=2, alpha=0.5)
ax2.set_title('AM-AM Characteristic')
ax2.set_xlabel('Input Magnitude')
ax2.set_ylabel('Output Magnitude')
ax2.grid(True)

# 3. AM-PM
ax3 = fig.add_subplot(grid[1, 0])
ax3.scatter(amp_in, phase_diff, s=2, alpha=0.5)
ax3.set_title('AM-PM Characteristic')
ax3.set_xlabel('Input Magnitude')
ax3.set_ylabel('Phase Change (rad)')
ax3.grid(True)

# 4. DDC Output Constellation
ax4 = fig.add_subplot(grid[1, 1])
ax4.scatter(ddc_i, ddc_q, s=2, alpha=0.5)
ax4.set_title('DDC Output Constellation')
ax4.set_xlabel('I')
ax4.set_ylabel('Q')
ax4.axis('equal')
ax4.grid(True)

# 5. Histogram of Output Magnitude
ax5 = fig.add_subplot(grid[2, 0])
ax5.hist(amp_out, bins=100, alpha=0.7, label='DDC Output')
ax5.hist(pa_with_dpd_aligned, bins=100, alpha=0.5, color='red', label='PA w/DPD (Aligned)')
ax5.set_title('Histogram of Output Magnitude')
ax5.set_xlabel('Output Magnitude')
ax5.set_ylabel('Count')
ax5.legend()
ax5.grid(True)

# 6. Time-domain overlay: All signals (I channel)
ax6 = fig.add_subplot(grid[2, 1])
ax6.plot(dpd_i_up[:n_show], label='DPD Input I')
ax6.plot(pa_with_dpd_aligned, label='PA Output w/DPD (Aligned)', color='red')
ax6.plot(ddc_i[:n_show], label='DDC Output I', alpha=0.7)
ax6.set_title('Time-Domain: Signal Comparison')
ax6.set_xlabel('Sample')
ax6.set_ylabel('Amplitude')
ax6.legend()
ax6.grid(True)

# 7. Normalized Input vs Output comparison (shape matching)
ax7 = fig.add_subplot(grid[3, 0])
norm_input_i = dpd_i_up[:n_show] / np.max(np.abs(dpd_i_up[:n_show]))
norm_output_i = ddc_i[:n_show] / np.max(np.abs(ddc_i[:n_show]))
norm_pa_dpd = pa_with_dpd_aligned / np.max(np.abs(pa_with_dpd_aligned))
ax7.plot(norm_input_i, label='Normalized Input')
ax7.plot(norm_pa_dpd, label='Normalized PA w/DPD', color='red')
ax7.plot(norm_output_i, label='Normalized DDC', alpha=0.7)
ax7.set_title('Shape Comparison (Normalized)')
ax7.set_xlabel('Sample')
ax7.set_ylabel('Normalized Amplitude')
ax7.legend()
ax7.grid(True)

# 8. Correlation scatter plot: Input vs PA Output
ax8 = fig.add_subplot(grid[3, 1])
ax8.scatter(norm_input_i, norm_pa_dpd, s=1, alpha=0.5, color='red')
ax8.set_title(f'Input vs PA Output Correlation (r={corr:.4f})')
ax8.set_xlabel('Normalized Input')
ax8.set_ylabel('Normalized PA Output w/DPD')
# Add ideal response line
lims = [
    np.min([ax8.get_xlim(), ax8.get_ylim()]),  
    np.max([ax8.get_xlim(), ax8.get_ylim()]),
]
ax8.plot(lims, lims, 'r-', alpha=0.5, label='Ideal (y=x)')
ax8.legend()
ax8.grid(True)

# 9. PA Output with DPD - Magnitude Analysis
ax9 = fig.add_subplot(grid[4, 0])
ax9.plot(pa_with_dpd_aligned, color='red')
ax9.axhline(y=np.max(pa_with_dpd), color='k', linestyle='--', label=f'Max: {np.max(pa_with_dpd):.2f}')
ax9.axhline(y=np.min(pa_with_dpd), color='k', linestyle=':', label=f'Min: {np.min(pa_with_dpd):.2f}')
ax9.axhline(y=0, color='gray', alpha=0.5)
ax9.set_title('PA Output with DPD Analysis (Aligned)')
ax9.set_xlabel('Sample')
ax9.set_ylabel('Amplitude')
ax9.legend()
ax9.grid(True)

# 10. DPD Input vs PA Output (Direct Comparison with Scaling)
ax10 = fig.add_subplot(grid[4, 1])
# Calculate gain/scale factor
scale_factor = np.mean(np.abs(pa_with_dpd_aligned)) / np.mean(np.abs(dpd_i_up[:n_show]))
scaled_input = dpd_i_up[:n_show] * scale_factor
ax10.plot(scaled_input, label=f'Scaled Input (×{scale_factor:.2f})')
ax10.plot(pa_with_dpd_aligned, color='red', label='PA Output w/DPD (Aligned)', alpha=0.7)
ax10.set_title(f'DPD Input vs PA Output (NMSE: {nmse_pa_dpd:.2f} dB)')
ax10.set_xlabel('Sample')
ax10.set_ylabel('Amplitude')
ax10.legend()
ax10.grid(True)

# Add performance metrics text box
metrics_text = f"Performance Metrics:\n" \
               f"NMSE (I): {nmse_i:.2f} dB\n" \
               f"NMSE (Q): {nmse_q:.2f} dB\n" \
               f"NMSE (DDC Total): {nmse_total:.2f} dB\n" \
               f"NMSE (Input vs PA): {nmse_pa_dpd:.2f} dB\n" \
               f"PA Range: [{np.min(pa_with_dpd):.2f}, {np.max(pa_with_dpd):.2f}]\n" \
               f"Alignment Shift: {shift}, Correlation: {corr:.4f}"
fig.text(0.02, 0.01, metrics_text, fontsize=10,
         bbox=dict(facecolor='white', alpha=0.7))

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()