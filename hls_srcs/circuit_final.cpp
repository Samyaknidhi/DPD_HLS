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

#define MAX_INPUT_BYTES 8192
#define DATA_LEN 8192
#define MAX_SYMBOLS 32768
#define INTERPOLATION_FACTOR 8
#define DECIM_FACTOR 8
#define DELAY_OFFSET 12 // Adjust as needed for your system

typedef ap_fixed<24,8> fixed_t;
typedef ap_fixed<24,8> data_t;
typedef ap_fixed<24,8> sample_type;
typedef ap_fixed<24,8> baseband_t;
typedef ap_fixed<32,16> adc_in_t;
typedef ap_fixed<32,16> adc_out_t;
typedef ap_fixed<24,8> data_ty;

void dual_adc_system(
    ap_fixed<32,16> I_analog_in[N], ap_fixed<32,16> Q_analog_in[N],
    ap_fixed<32,16> I_digital_out[N], ap_fixed<32,16> Q_digital_out[N]
) {
#pragma HLS INTERFACE ap_memory port=I_analog_in
#pragma HLS INTERFACE ap_memory port=Q_analog_in
#pragma HLS INTERFACE ap_memory port=I_digital_out
#pragma HLS INTERFACE ap_memory port=Q_digital_out
#pragma HLS INTERFACE ap_ctrl_hs port=return

    const ap_fixed<32,16> V_REF = 1.0;
    const ap_fixed<32,16> V_MIN = -V_REF;
    const ap_fixed<32,16> V_MAX = V_REF;

    // ADC quantization parameters
    const int ADC_LEVELS = (1 << W);  // 65536 levels for 16-bit
    const ap_fixed<32,16> ADC_STEP = (V_MAX - V_MIN) / (ADC_LEVELS - 1);  // ~3.05e-5V per step

    for (int i = 0; i < N; i++) {
    #pragma HLS PIPELINE II=1

        // I channel: True ADC quantization
        ap_fixed<32,16> clamped_I = (I_analog_in[i] > V_MAX) ? V_MAX :
                                   ((I_analog_in[i] < V_MIN) ? V_MIN : I_analog_in[i]);

        // Step 1: Convert to quantization level (0 to 65535)
        ap_fixed<32,17> normalized_I = (clamped_I - V_MIN) / ap_fixed<32,17>(ADC_STEP);
        int quantized_level_I = (int)(normalized_I + ap_fixed<32,17>(0.5));  // Round to nearest level

        // Step 2: Clamp to valid ADC range
        if (quantized_level_I < 0) quantized_level_I = 0;
        if (quantized_level_I >= ADC_LEVELS) quantized_level_I = ADC_LEVELS - 1;

        // Step 3: Convert back to voltage (this is the "digital" representation)
        // This maintains signal levels while providing quantization effects
        I_digital_out[i] = V_MIN + (ap_fixed<32,17>(quantized_level_I) * ap_fixed<32,17>(ADC_STEP));

        // Q channel: Same process
        ap_fixed<32,16> clamped_Q = (Q_analog_in[i] > V_MAX) ? V_MAX :
                                   ((Q_analog_in[i] < V_MIN) ? V_MIN : Q_analog_in[i]);

        ap_fixed<32,17> normalized_Q = (clamped_Q - V_MIN) / ap_fixed<32,17>(ADC_STEP);
        int quantized_level_Q = (int)(normalized_Q + ap_fixed<32,17>(0.5));

        if (quantized_level_Q < 0) quantized_level_Q = 0;
        if (quantized_level_Q >= ADC_LEVELS) quantized_level_Q = ADC_LEVELS - 1;

        Q_digital_out[i] = V_MIN + (ap_fixed<32,17>(quantized_level_Q) * ap_fixed<32,17>(ADC_STEP));

        // Debug: Show quantization effect
        #ifndef __SYNTHESIS__
        if (i < 10 || i > N-10) {
            printf("Sample %d: I_analog=%f -> I_digital=%f (level=%d, diff=%f) | Q_analog=%f -> Q_digital=%f (level=%d, diff=%f)\n",
                   i, clamped_I.to_double(), I_digital_out[i].to_double(), quantized_level_I,
                   (I_digital_out[i] - clamped_I).to_double(),
                   clamped_Q.to_double(), Q_digital_out[i].to_double(), quantized_level_Q,
                   (Q_digital_out[i] - clamped_Q).to_double());
        }
        #endif
    }
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
    bool adapt // true for adaptation, false for transmit/verification
) {
    static ap_uint<8> nco_phase = 0;
    static bool nco_initialized = false;
    if (!nco_initialized) {
        nco_phase = 0;  // Initialize once
        nco_initialized = true;
    }
    // Static DPD weights (persist between calls)
    static ccoef_t w[K][MEMORY_DEPTH] = {0};
    static bool weights_initialized = false;

    if (!weights_initialized) {
        for (int m = 0; m < MEMORY_DEPTH; ++m) {
            w[0][m].real = 1.0; // Linear passthrough
            w[0][m].imag = 0.0;
        }
        weights_initialized = true;
    }
    // 1. Pulse shaping (feedforward)
    pulse_shape(i_symbols, q_symbols, i_psf, q_psf);

    coef_t mu = adapt ? 0.0000001 : 0.0; // Adaptation only if adapt==true

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
        dpd(i_in, q_in, i_ref, q_ref, w, 0.0, &z_i, &z_q); // Adapt if mu>0
        dpd_i[n] = z_i;
        dpd_q[n] = z_q;

        data_ty dac_i, dac_q;
        dac_multibit_with_select(z_i, dac_i, 0);
        dac_multibit_with_select(z_q, dac_q, 1);
        dac_i_arr[n] = dac_i;
        dac_q_arr[n] = dac_q;

        data_t i_mod_fixed = data_t(dac_i) * data_t(1.0/128.0);
        data_t q_mod_fixed = data_t(dac_q) * data_t(1.0/128.0);

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
        const sample_type DUC_SCALE = sample_type(1.0);  // Adjust based on your PA requirements
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

    // 5. DDC Stage (Feedback path) - COMPLETELY FIXED
    static rf_sample_t ddc_in[DATA_LEN * INTERPOLATION_FACTOR];
    for (int i = 0; i < DATA_LEN * INTERPOLATION_FACTOR; ++i) {
        ddc_in[i] = rf_sample_t(amp_out_i[i] * data_t(1.0));
    }

    ap_uint<32> ddc_freq_word = 0x40000000;  // Same frequency as DUC

    // FIXED: Reasonable DDC gain (was 25000, now much smaller)
    ap_fixed<16,8> ddc_gain = 40.0;  // Reduced by 1000x - will be further scaled in DDC

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
    #endif

    // 6. ADC Stage (Final Output)
    const int ADC_LEN = (DATA_LEN * INTERPOLATION_FACTOR) / DECIM_FACTOR;
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


        // 9. DPD Adaptation (Indirect Learning) -- Only if adapt==true
    if (adapt) {
            double max_abs_val = 1e-9;
            for (int i = 0; i < ADC_LEN; ++i) {
                double mag = std::sqrt(double(i_psf_fb[i])*double(i_psf_fb[i]) + double(q_psf_fb[i])*double(q_psf_fb[i]));
                if (mag > max_abs_val) max_abs_val = mag;
            }
            for (int i = 0; i < DATA_LEN; ++i) {
                double mag = std::sqrt(double(dpd_i[i])*double(dpd_i[i]) + double(dpd_q[i])*double(dpd_q[i]));
                if (mag > max_abs_val) max_abs_val = mag;
            }
            double peak_norm_factor = (max_abs_val > 1e-9) ? (1.0 / max_abs_val) : 1.0;
            coef_t mu = 0.0000001;

            for (int n = 0; n < DATA_LEN && n < ADC_LEN; ++n) {
                data_t i_in_raw[MEMORY_DEPTH] = {0}; data_t q_in_raw[MEMORY_DEPTH] = {0};
                for (int m = 0; m < MEMORY_DEPTH; ++m) {
                    int idx = n - m;
                    if (idx >= 0) { i_in_raw[m] = data_t(i_psf_fb[idx]); q_in_raw[m] = data_t(q_psf_fb[idx]); }
                }
                int ref_idx = (n >= DELAY_OFFSET) ? (n - DELAY_OFFSET) : 0;
                data_t i_ref_raw = (ref_idx < DATA_LEN) ? dpd_i[ref_idx] : data_t(0);
                data_t q_ref_raw = (ref_idx < DATA_LEN) ? dpd_q[ref_idx] : data_t(0);

                data_t i_in_norm[MEMORY_DEPTH], q_in_norm[MEMORY_DEPTH];
                for(int m=0; m<MEMORY_DEPTH; ++m) {
                    i_in_norm[m] = data_t(double(i_in_raw[m]) * peak_norm_factor);
                    q_in_norm[m] = data_t(double(q_in_raw[m]) * peak_norm_factor);
                }
                data_t i_ref_norm = data_t(double(i_ref_raw) * peak_norm_factor);
                data_t q_ref_norm = data_t(double(q_ref_raw) * peak_norm_factor);

                data_t z_i, z_q;
                dpd(i_in_norm, q_in_norm, i_ref_norm, q_ref_norm, w, mu, &z_i, &z_q);

                // Debug the CORRECT adaptation
                #ifndef __SYNTHESIS__
                if (n < 5) {
                    printf("DPD_ADAPT[%d]: PA_output_in=(%f,%f), PA_input_ref=(%f,%f), DPD_out=(%f,%f)\n",
                           n, i_in_norm[0].to_double(), q_in_norm[0].to_double(),
                           i_ref_norm.to_double(), q_ref_norm.to_double(),
                           z_i.to_double(), z_q.to_double());

                    data_t err_i = i_ref_norm - z_i;
                    data_t err_q = q_ref_norm - z_q;
                    printf("DPD_ADAPT[%d]: Error=(%f,%f), |Error|=%f\n",
                           n, err_i.to_double(), err_q.to_double(),
                           std::sqrt(err_i.to_double()*err_i.to_double() + err_q.to_double()*err_q.to_double()));
                }
                #endif
            }

            // Debug weight evolution after adaptation
            #ifndef __SYNTHESIS__
            printf("DPD Weights after adaptation:\n");
            for (int k = 0; k < K && k < 3; ++k) {
                printf("  Weight[%d][0]: %f + j%f\n", k, w[k][0].real.to_double(), w[k][0].imag.to_double());
            }
            #endif
        }
}
