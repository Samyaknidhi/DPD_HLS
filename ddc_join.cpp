#include "ap_int.h"
#include <ap_fixed.h>
#include "hls_stream.h"
#include "hls_math.h"
#include <iostream>

// Configuration parameters
#define DECIM_FACTOR 8            // Decimation factor
#define FIR_TAPS 63               // Number of FIR filter taps
#define NCO_LUT_SIZE 1024         // NCO LUT size

// Fixed-point type definitions
typedef ap_fixed<16,8> rf_sample_t;       // 16-bit RF input samples
typedef ap_fixed<18,2> filter_coeff_t;    // Filter coefficient type
typedef ap_fixed<24,8> filter_accum_t;    // Accumulator with growth for FIR filters
typedef ap_fixed<24,8> baseband_t;        // Baseband I/Q output samples
typedef ap_uint<10> phase_t;              // NCO phase accumulator index

// FIXED: Use your existing LUT files with correct type casting
const filter_accum_t sine_lut[NCO_LUT_SIZE] = {
    #include "sin_lut.h"  // Your existing file - values will be cast to filter_accum_t
};

const filter_accum_t cosine_lut[NCO_LUT_SIZE] = {
    #include "cos_lut.h"  // Your existing file - values will be cast to filter_accum_t
};

// Pre-computed low-pass filter coefficients (symmetric FIR)
static const filter_coeff_t lpf_coeffs[FIR_TAPS] = {
    -0.0017, -0.0020, -0.0015, 0.0000, 0.0026, 0.0057, 0.0086, 0.0106, 0.0110,
    0.0094, 0.0058, 0.0006, -0.0053, -0.0110, -0.0155, -0.0180, -0.0178, -0.0145,
    -0.0082, -0.0000, 0.0086, 0.0163, 0.0216, 0.0234, 0.0212, 0.0151, 0.0057,
    -0.0053, -0.0160, -0.0247, -0.0298, -0.0302, -0.0251, -0.0147, 0.0000,
    0.0171, 0.0342, 0.0490, 0.0596, 0.0644, 0.0626, 0.0544, 0.0411, 0.0245,
    0.0072, -0.0082, -0.0194, -0.0249, -0.0245, -0.0188, -0.0094, 0.0012,
    0.0106, 0.0171, 0.0198, 0.0184, 0.0135, 0.0067, 0.0000, 0.0000, -0.0053, 0.0000, 0.0000
};

/**
 * FIXED: Digital Downconverter with proper scaling and types
 */
void ddc_demodulator(
    rf_sample_t rf_in[],
    baseband_t i_out[],
    baseband_t q_out[],
    int num_samples,
    ap_uint<32> freq_word,
    ap_fixed<16,8> gain
) {
    #pragma HLS INTERFACE ap_memory port=rf_in
    #pragma HLS INTERFACE ap_memory port=i_out
    #pragma HLS INTERFACE ap_memory port=q_out
    #pragma HLS INTERFACE s_axilite port=num_samples
    #pragma HLS INTERFACE s_axilite port=freq_word
    #pragma HLS INTERFACE s_axilite port=gain
    #pragma HLS INTERFACE ap_ctrl_hs port=return
    #pragma HLS ARRAY_PARTITION variable=lpf_coeffs complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sine_lut cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=cosine_lut cyclic factor=4 dim=1

    // FIXED: Static filter buffers that persist between calls
    static filter_accum_t i_buffer[FIR_TAPS];
    static filter_accum_t q_buffer[FIR_TAPS];
    static bool buffers_initialized = false;

    #pragma HLS ARRAY_PARTITION variable=i_buffer cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=q_buffer cyclic factor=8

    // FIXED: Initialize buffers only once
    if (!buffers_initialized) {
        for (int i = 0; i < FIR_TAPS; i++) {
            #pragma HLS UNROLL
            i_buffer[i] = 0;
            q_buffer[i] = 0;
        }
        buffers_initialized = true;
    }

    // FIXED: Static NCO phase accumulator (persists between calls)
    static ap_uint<32> phase_acc = 0;

    // Track output sample count
    int out_sample_idx = 0;
    int decim_count = 0;

    // Calculate expected output samples
    const int expected_outputs = num_samples / DECIM_FACTOR;


    // Main processing loop
    for (int n = 0; n < num_samples; n++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=8192 max=65536 avg=32768

        // Bounds check
        if (n >= num_samples) break;

        // Get input sample
        rf_sample_t rf_sample = rf_in[n];

        // Update NCO phase
        phase_acc += freq_word;

        // FIXED: Proper LUT indexing with bounds checking
        phase_t table_idx = phase_acc >> (32 - 10);  // Use top 10 bits
        if (table_idx >= NCO_LUT_SIZE) table_idx = NCO_LUT_SIZE - 1;

        // FIXED: Use correct types - cast LUT values to filter_accum_t
        filter_accum_t sin_val = sine_lut[table_idx];
        filter_accum_t cos_val = cosine_lut[table_idx];

        // Debug early samples
        if (n < 5) {
            std::cout << "[n=" << n << "] rf_sample = " << rf_sample.to_double()
                     << ", sin_val = " << sin_val.to_double()
                     << ", cos_val = " << cos_val.to_double() << std::endl;
        }

        // Quadrature mixing (downconversion)
        filter_accum_t i_mixed = rf_sample * cos_val;
        filter_accum_t q_mixed = rf_sample * (-sin_val);  // Negative for downconversion

        if (n < 5) {
            std::cout << "[n=" << n << "] i_mixed = " << i_mixed.to_double()
                     << ", q_mixed = " << q_mixed.to_double() << std::endl;
        }

        // Shift samples through delay line
        for (int i = FIR_TAPS-1; i > 0; i--) {
            #pragma HLS UNROLL factor=8
            i_buffer[i] = i_buffer[i-1];
            q_buffer[i] = q_buffer[i-1];
        }

        // Insert new mixed samples
        i_buffer[0] = i_mixed;
        q_buffer[0] = q_mixed;

        // FIXED: Decimating filter with bounds checking
        if (decim_count == 0) {
            // Compute FIR filter outputs
            filter_accum_t i_acc = 0;
            filter_accum_t q_acc = 0;

            for (int i = 0; i < FIR_TAPS; i++) {
                #pragma HLS UNROLL factor=8
                i_acc += i_buffer[i] * lpf_coeffs[i];
                q_acc += q_buffer[i] * lpf_coeffs[i];
            }

            if (out_sample_idx < 5) {
                std::cout << "[out_sample_idx=" << out_sample_idx << "] i_acc = " << i_acc.to_double()
                         << ", q_acc = " << q_acc.to_double() << std::endl;
            }

            // FIXED: Apply reasonable gain scaling and bounds check
            if (out_sample_idx < expected_outputs) {
                i_out[out_sample_idx] = i_acc * filter_accum_t(gain);
                q_out[out_sample_idx] = q_acc * filter_accum_t(gain);

                if (out_sample_idx < 5) {
                    std::cout << "[out_sample_idx=" << out_sample_idx << "] final i_out = "
                             << i_out[out_sample_idx].to_double()
                             << ", final q_out = " << q_out[out_sample_idx].to_double() << std::endl;
                }

                out_sample_idx++;
            }
        }

        // Update decimation counter
        decim_count = (decim_count + 1) % DECIM_FACTOR;
    }

    // Debug output
    #ifndef __SYNTHESIS__
    std::cout << "DDC: Processed " << num_samples << " input samples, generated "
              << out_sample_idx << " output samples (expected " << expected_outputs << ")" << std::endl;
    std::cout << "DDC: Applied gain scaling factor: " << gain.to_double() << std::endl;
    #endif
}
