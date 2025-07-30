	#include <ap_int.h>
#include <ap_fixed.h>
#include <iostream>
#include <cmath>
#include "conste_join.h"
#include "psf_join.h"
#include "dpd_joiner.h"
#include "dac_joiner.h"
#include "qm_joiner.h"
#include "duc_joiner.h"
#include "pa_joiner.h"
#include "ddc_joiner.h"
#include "hls_stream.h"
#include "adc_joiner.h"
#include <complex> // <-- Make sure this include is present
#define MAX_INPUT_BYTES 8192
#define DATA_LEN 8192
#define MAX_SYMBOLS 32768
#define INTERPOLATION_FACTOR 8
#define DECIM_FACTOR 8
#define DELAY_OFFSET 1252 // Adjust as needed for your system

typedef ap_fixed<24,8> fixed_t;
typedef ap_fixed<24,8> data_t;
typedef ap_fixed<24,8> sample_type;
typedef ap_fixed<24,8> baseband_t;
typedef ap_fixed<32,16> adc_in_t;
typedef ap_fixed<32,16> adc_out_t;
typedef ap_fixed<24,8> data_ty;

//================================================================================
// NEW: AUTOMATIC DELAY FINDER FUNCTION
//================================================================================
typedef std::complex<data_t> complex_data_t;
const int SYNC_SEARCH_WINDOW = 200;
const int SYNC_CORR_LENGTH = 2048;

int find_best_delay(
    data_t ref_i[SYNC_CORR_LENGTH], data_t ref_q[SYNC_CORR_LENGTH],
    fixed_t fb_i[], fixed_t fb_q[], int fb_len
) {
    long long max_correlation = 0;
    int best_delay = 0;

    delay_search_loop: for (int d = 0; d < SYNC_SEARCH_WINDOW; ++d) {
        #pragma HLS PIPELINE
        long long current_correlation = 0;
        correlation_loop: for (int i = 0; i < SYNC_CORR_LENGTH; ++i) {
            if ((i + d) < fb_len) {
                complex_data_t ref_sample(ref_i[i], ref_q[i]);
                complex_data_t fb_sample(data_t(fb_i[i + d]), data_t(fb_q[i + d]));
                complex_data_t product = std::conj(ref_sample) * fb_sample;
                current_correlation += (long long)(product.real() * product.real() + product.imag() * product.imag());
            }
        }
        if (current_correlation > max_correlation) {
            max_correlation = current_correlation;
            best_delay = d;
        }
    }
    return best_delay;
}


void circuit_final(
    ap_fixed<16,8> input_bytes[MAX_INPUT_BYTES],
    int num_bits,
    sample_type duc_out[DATA_LEN * INTERPOLATION_FACTOR],
    fixed_t i_symbols[MAX_SYMBOLS],
	fixed_t q_symbols[MAX_SYMBOLS],
	fixed_t i_psf[DATA_LEN],
	fixed_t q_psf[DATA_LEN],
	data_t dpd_i[DATA_LEN],
	data_t dpd_q[DATA_LEN],
	data_ty dac_i_arr[DATA_LEN],
	data_ty dac_q_arr[DATA_LEN],
	data_t qm_out_buf[DATA_LEN],
	data_t amp_out_i[DATA_LEN * INTERPOLATION_FACTOR],
	data_t amp_out_q[DATA_LEN * INTERPOLATION_FACTOR],
	data_t amp_magnitude[DATA_LEN * INTERPOLATION_FACTOR],
	data_t amp_gain_lin[DATA_LEN * INTERPOLATION_FACTOR],
	data_t amp_gain_db[DATA_LEN * INTERPOLATION_FACTOR],
	baseband_t ddc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
	baseband_t ddc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
	adc_out_t adc_i_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
	adc_out_t adc_q_out[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
	fixed_t i_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
	fixed_t q_psf_fb[(DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR],
	ccoef_t w[K][MEMORY_DEPTH],
    bool adapt // true for adaptation, false for transmit/verification
) {
    static ap_uint<8> nco_phase = 0;
    static bool nco_initialized = false;
    if (!nco_initialized) {
        nco_phase = 0;  // Initialize once
        nco_initialized = true;
    }
    // Static DPD weights (persist between calls)
    /*static ccoef_t w[K][MEMORY_DEPTH] = {0};
    static bool weights_initialized = false;

    if (!weights_initialized) {
        for (int m = 0; m < MEMORY_DEPTH; ++m) {
            w[0][m].real = 1.0; // Linear passthrough
            w[0][m].imag = 0.0;
        }
        weights_initialized = true;
    }*/
    static bool delay_has_been_calculated = false;
    static int internal_delay_offset = 0; // Will store the calculated delay

    // 1. Pulse shaping (feedforward)
    pulse_shape(i_symbols, q_symbols, i_psf, q_psf);
    // ===================================================================
        // FINAL FIX: Normalize the pulse-shaped signal BEFORE it enters the DPD/DAC chain.
        // This prevents clipping in the DAC and ensures the PA is driven correctly.
        // ===================================================================
        data_t max_psf_val = 1e-9;
        for (int i = 0; i < DATA_LEN; i++) {
            #pragma HLS PIPELINE
            data_t i_abs = std::abs(float(i_psf[i]));
            data_t q_abs = std::abs(float(q_psf[i]));
            if (i_abs > max_psf_val) max_psf_val = i_abs;
            if (q_abs > max_psf_val) max_psf_val = q_abs;
        }

        // Apply a scaling factor to bring the peak magnitude to just under 1.0
        if (max_psf_val > 1.0) {
            data_t psf_scale = data_t(0.95) / max_psf_val;
            for (int i = 0; i < DATA_LEN; i++) {
                #pragma HLS PIPELINE
                i_psf[i] *= psf_scale;
                q_psf[i] *= psf_scale;
            }
        }
        // ===================================================================

    coef_t mu = adapt ? 0.0003 : 0.0; // Adaptation only if adapt==true

    ap_uint<8> phase_inc = 2;

    // 2. DPD (transmit path)
    for (int n = 0; n < DATA_LEN; ++n) {
        data_t i_in[MEMORY_DEPTH] = {0};
        data_t q_in[MEMORY_DEPTH] = {0};
        for (int m = 0; m < MEMORY_DEPTH; ++m) {
            int idx = n - m;
            if (idx >= 0) {
                i_in[m] = (data_t)i_psf[idx];
                q_in[m] = (data_t)q_psf[idx];
            } else {
                i_in[m] = 0;
                q_in[m] = 0;
            }
        }
        data_t i_ref = i_psf[n];
        data_t q_ref = q_psf[n];

        data_t z_i, z_q;
        dpd(i_in, q_in, data_t(0), data_t(0), w, 0.0, &z_i, &z_q); // Adapt if mu>0
        dpd_i[n] = z_i;
        dpd_q[n] = z_q;

        data_ty dac_i, dac_q;
        dac_multibit_with_select(z_i, dac_i, 0);
        dac_multibit_with_select(z_q, dac_q, 1);
        dac_i_arr[n] = dac_i;
        dac_q_arr[n] = dac_q;

        data_t i_mod_fixed = data_t(dac_i);
        data_t q_mod_fixed = data_t(dac_q);

        data_t cos_lo, sin_lo;
        nco(nco_phase, phase_inc, cos_lo, sin_lo);

        data_t qm_out = digital_qm(i_mod_fixed, q_mod_fixed, cos_lo, sin_lo);
        qm_out_buf[n] = qm_out;
    }

    // Replace the DUC section with this:

        // 3. DUC Stage - FIXED
        static sample_type duc_i_input[DATA_LEN];
        static sample_type duc_q_input[DATA_LEN];

        // Prepare DUC inputs
        for (int i = 0; i < DATA_LEN; ++i) {
            duc_i_input[i] = sample_type(qm_out_buf[i]);
            duc_q_input[i] = sample_type(0);  // Single sideband after QM
        }

        ap_uint<32> freq_control_word = 0x40000000;  // Quarter sample rate
        ap_uint<1> enable = 1;

        // FIXED: Use batch processing (1 call instead of DATA_LEN calls)
        digital_upconverter_batch(
            duc_i_input,
            duc_q_input,
            duc_out,  // Direct output to duc_out array
            freq_control_word,
            enable
        );

        // FIXED: More reasonable scaling
        const sample_type DUC_SCALE = sample_type(1.5);  // Adjust based on your PA requirements
        for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
            duc_out[i] = duc_out[i] * DUC_SCALE;
        }

        // Debug DUC output
        #ifndef __SYNTHESIS__
        printf("DUC: Input samples=%d, Output samples=%d, Scale factor=%f\n",
               DATA_LEN, DATA_LEN * INTERPOLATION_FACTOR, DUC_SCALE.to_double());
        printf("DUC: First few outputs: %f, %f, %f\n",
               duc_out[0].to_double(), duc_out[1].to_double(), duc_out[2].to_double());
        #endif

    // 4. Amplifier Stage
    for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
        data_t local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db;
        saleh_amplifier(
            data_t(duc_out[i]),
            data_t(0),
            local_amp_i, local_amp_q, local_amp_mag, local_amp_gain_lin, local_amp_gain_db
        );
        amp_out_i[i] = local_amp_i;
        amp_out_q[i] = local_amp_q;
        amp_magnitude[i] = local_amp_mag;
        amp_gain_lin[i] = local_amp_gain_lin;
        amp_gain_db[i] = local_amp_gain_db;
    }

    /// Replace DDC section with this:

    /*// 5. DDC Stage (Feedback path) - COMPLETELY FIXED
    static rf_sample_t ddc_in[DATA_LEN * INTERPOLATION_FACTOR];
    for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
        ddc_in[i] = rf_sample_t(amp_out_i[i] * data_t(1.0));
    }

    ap_uint<32> ddc_freq_word = 0x40000000;  // Same frequency as DUC

    // FIXED: Reasonable DDC gain (was 25000, now much smaller)
    ap_fixed<16,8> ddc_gain = 0.12;  // Reduced by 1000x - will be further scaled in DDC

    // DDC automatically handles decimation: 65536 in -> 8192 out
    ddc_demodulator(
        ddc_in,                              // <- Input: 65536 RF samples
        ddc_i_out,                          // <- Output: 8192 I samples
        ddc_q_out,                          // <- Output: 8192 Q samples
        DATA_LEN * INTERPOLATION_FACTOR,    // <- Input size: 65536
        ddc_freq_word,
        ddc_gain
    );

    // Debug DDC operation
    #ifndef __SYNTHESIS__
    printf("DDC: Input samples=%d, Expected output samples=%d\n",
           DATA_LEN * INTERPOLATION_FACTOR, (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR);
    printf("DDC: Applied gain=%f\n", ddc_gain.to_double());
    printf("DDC: First few I outputs: %f, %f, %f\n",
           ddc_i_out[0].to_double(), ddc_i_out[1].to_double(), ddc_i_out[2].to_double());
    printf("DDC: First few Q outputs: %f, %f, %f\n",
           ddc_q_out[0].to_double(), ddc_q_out[1].to_double(), ddc_q_out[2].to_double());
    #endif*/

    // 5. Signal Conditioning after PA (Feedback Path)
        static i_fir_state_t i_fir_state_fb;
        static q_fir_state_t q_fir_state_fb;

        int fb_out_idx = 0;
        const data_t FEEDBACK_GAIN = 0.5;

        // Process the full, high-rate signal from the amplifier model
        for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
            #pragma HLS PIPELINE

            bool sample_is_valid;

            // Call the new, safe dual-path function once per loop iteration
            fir_decim_dual_path(
                amp_out_i[i],
                amp_out_q[i],
                ddc_i_out[fb_out_idx],
                ddc_q_out[fb_out_idx],
                i_fir_state_fb,
                q_fir_state_fb,
                sample_is_valid
            );

            // When the filter provides a valid output, apply gain and advance the index
            if (sample_is_valid) {
                ddc_i_out[fb_out_idx] *= FEEDBACK_GAIN;
                ddc_q_out[fb_out_idx] *= FEEDBACK_GAIN;
                fb_out_idx++;
            }
        }

        #ifndef __SYNTHESIS__
        printf("Feedback Path: Processed %d PA samples, generated %d decimated samples.\n",
               DATA_LEN * INTERPOLATION_FACTOR, fb_out_idx);
        printf("Feedback Path: First few I outputs: %f, %f, %f\n",
               ddc_i_out[0].to_double(), ddc_i_out[1].to_double(), ddc_i_out[2].to_double());
        printf("Feedback Path: First few Q outputs: %f, %f, %f\n",
               ddc_q_out[0].to_double(), ddc_q_out[1].to_double(), ddc_q_out[2].to_double());
        #endif

    // =========================================================================
    // >>>>>>>>>> END OF REPLACEMENT BLOCK <<<<<<<<<<
    // =========================================================================
        const int ADC_LEN = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;
            data_t max_adc_in_val = 1e-9;
            for (int i = 0; i < ADC_LEN; i++) {
                #pragma HLS PIPELINE
                data_t i_abs = std::abs(float(ddc_i_out[i]));
                data_t q_abs = std::abs(float(ddc_q_out[i]));
                if (i_abs > max_adc_in_val) max_adc_in_val = i_abs;
                if (q_abs > max_adc_in_val) max_adc_in_val = q_abs;
            }

            if (max_adc_in_val > 1.0) {
                data_t adc_scale = data_t(0.95) / max_adc_in_val;
                for (int i = 0; i < ADC_LEN; i++) {
                    #pragma HLS PIPELINE
                    ddc_i_out[i] *= adc_scale;
                    ddc_q_out[i] *= adc_scale;
                }
            }

    // 6. ADC Stage (Final Output)
    //const int ADC_LEN = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;
    static adc_in_t adc_i_in[ADC_LEN], adc_q_in[ADC_LEN];
    for (int i = 0; i < ADC_LEN; ++i) {
        adc_i_in[i] = adc_in_t(ddc_i_out[i] * adc_in_t(1.0));
        adc_q_in[i] = adc_in_t(ddc_q_out[i] * adc_in_t(1.0));
    }
    dual_adc_system(adc_i_in, adc_q_in, adc_i_out, adc_q_out);

    // 7. PSF after ADC (Feedback path)
    static fixed_t i_psf_fb_in[ADC_LEN], q_psf_fb_in[ADC_LEN];
    for (int i = 0; i < ADC_LEN; ++i) {
        i_psf_fb_in[i] = fixed_t(adc_i_out[i]);
        q_psf_fb_in[i] = fixed_t(adc_q_out[i]);
    }
    pulse_shape(i_psf_fb_in, q_psf_fb_in, i_psf_fb, q_psf_fb);

        // This block runs only on the first call where adapt is true.
        if (adapt && !delay_has_been_calculated) {
            #ifndef __SYNTHESIS__
            printf("--- Running one-time delay synchronization... ---\n");
            #endif

            internal_delay_offset = find_best_delay(
                dpd_i,
                dpd_q,
                i_psf_fb,
                q_psf_fb,
                (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR
            );
            delay_has_been_calculated = true;

            #ifndef __SYNTHESIS__
            printf(">>> Auto-calculated delay offset: %d <<<\n\n", internal_delay_offset);
            #endif
        }

    // 8. Normalization of feedback PSF to match feedforward PSF RMS
    /*double ff_rms = 0, fb_rms = 0;
    for (int i = 0; i < DATA_LEN; ++i) {
        ff_rms += double(i_psf[i]) * double(i_psf[i]);
    }
    for (int i = 0; i < ADC_LEN; ++i) {
        fb_rms += double(i_psf_fb[i]) * double(i_psf_fb[i]);
    }
    ff_rms = std::sqrt(ff_rms / DATA_LEN);
    fb_rms = std::sqrt(fb_rms / ADC_LEN);

    double norm_factor = (fb_rms > 1e-12) ? (ff_rms / fb_rms) : 1.0;

    for (int i = 0; i < ADC_LEN; ++i) {
        i_psf_fb[i] = fixed_t(double(i_psf_fb[i]) * norm_factor);
        q_psf_fb[i] = fixed_t(double(q_psf_fb[i]) * norm_factor);
    }*/

    // In circuit_final.cpp

        if (adapt) {
                // 1. Find the peak absolute value from the feedback signal buffer.
                data_t max_fb_val = 1e-9;
                for (int i = 0; i < ADC_LEN; i++) {
                    #pragma HLS PIPELINE
                    data_t i_abs = std::abs(float(i_psf_fb[i])); // Use hls::abs for safety
                    data_t q_abs = std::abs(float(q_psf_fb[i]));
                    if (i_abs > max_fb_val) max_fb_val = i_abs;
                    if (q_abs > max_fb_val) max_fb_val = q_abs;
                }

                // 2. Find the peak absolute value of the TARGET reference signal (i_psf)
                data_t max_ref_val = 1e-9;
                for (int i = 0; i < DATA_LEN; i++) {
                    #pragma HLS PIPELINE
                    data_t i_abs = std::abs(float(i_psf[i])); // Use hls::abs
                    if (i_abs > max_ref_val) max_ref_val = i_abs;
                }

                // 3. Calculate the gain correction factor to match relative power
                data_t gain_correction_factor = max_ref_val / max_fb_val;

                data_t training_scale = data_t(1.0) / max_ref_val;

                #ifndef __SYNTHESIS__
                printf("Normalization: max_ref_val=%f, max_fb_val=%f, gain_correction=%f, training_scale=%f\n",
                       max_ref_val.to_double(), max_fb_val.to_double(), gain_correction_factor.to_double(), training_scale.to_double());
                #endif

                // Set a stable learning rate
                coef_t mu = 0.0003; // A modest mu is good practice

                // 4. Main adaptation loop with corrected scaling
                for (int n = 0; n < DATA_LEN; ++n) {
                    #pragma HLS PIPELINE
                    data_t i_in_feedback[MEMORY_DEPTH] = {0};
                    data_t q_in_feedback[MEMORY_DEPTH] = {0};

                    // Build the training input vector
                    for (int m = 0; m < MEMORY_DEPTH; ++m) {
                        int idx = n - m;
                        if (idx >= 0 && idx < ADC_LEN) {
                            // Apply gain correction AND the new training scale
                            i_in_feedback[m] = data_t(i_psf_fb[idx]) * gain_correction_factor * training_scale;
                            q_in_feedback[m] = data_t(q_psf_fb[idx]) * gain_correction_factor * training_scale;
                        }
                    }

                    // Get the delayed reference target
                    int ref_idx = (n >= internal_delay_offset) ? (n - internal_delay_offset) : 0;
                    data_t i_ref_base = (ref_idx < DATA_LEN) ? i_psf[ref_idx] : data_t(0);
                    data_t q_ref_base = (ref_idx < DATA_LEN) ? q_psf[ref_idx] : data_t(0);

                    // Apply the NEW training scale to the target as well
                    data_t i_ref_target = i_ref_base * training_scale;
                    data_t q_ref_target = q_ref_base * training_scale;

                    data_t z_i, z_q;

                    // Call DPD to update weights using the now-stable, scaled signals
                    dpd(i_in_feedback, q_in_feedback, i_ref_target, q_ref_target, w, mu, &z_i, &z_q);

                    #ifndef __SYNTHESIS__
            	        if (n % (DATA_LEN / 8) == 0) {
            	            data_t err_i = i_ref_target - z_i;
            	            data_t err_q = q_ref_target - z_q;
            	            double err_mag = std::sqrt(float(err_i.to_double() * err_i.to_double() + err_q.to_double() * err_q.to_double()));
            	            printf("Adaptation Step [%d/%d]: |Normalized Error| = %f\n", n, DATA_LEN, err_mag);
            	        }
            	    #endif
            	}
        // ======================================================================
        // >>>>>>>>>> ADDED DEBUG PRINTS <<<<<<<<<<
        // ======================================================================
        #ifndef __SYNTHESIS__
        printf("\n--- DPD Weights After Adaptation ---\n");
        for (int k = 0; k < K; ++k) {
            for (int m = 0; m < MEMORY_DEPTH; ++m) {
                // Print only the most significant weights to avoid clutter
                if (k < 4 && m < 2) {
                    printf("  w[%d][%d]: %s%f + j*(%s%f)\n", k, m,
                        (w[k][m].real < 0 ? "" : " "), w[k][m].real.to_double(),
                        (w[k][m].imag < 0 ? "" : " "), w[k][m].imag.to_double());
                }
            }
        }
        printf("------------------------------------\n\n");
        #endif
    }
}







