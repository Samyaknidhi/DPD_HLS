#include "hls_math.h"
#include <iostream>
#include "ddc_joiner.h"

// Use the same coefficients as before
static const ap_fixed<18,2> lpf_coeffs[FIR_TAPS] = {
    -0.0408, -0.0480, -0.0360,  0.0000,  0.0624,  0.1369,  0.2065,  0.2546,  0.2642,
     0.2257,  0.1393,  0.0144, -0.1273, -0.2642, -0.3723, -0.4323, -0.4275, -0.3482,
    -0.1970,  0.0000,  0.2065,  0.3915,  0.5188,  0.5620,  0.5092,  0.3627,  0.1369,
    -0.1273, -0.3843, -0.5933, -0.7157, -0.7253, -0.6029, -0.3531,  0.0000,  0.4107,
     0.8214,  1.1769,  1.4314,  1.5467,  1.5034,  1.3065,  0.9872,  0.5884,  0.1730,
    -0.1970, -0.4660, -0.5981, -0.5884, -0.4515, -0.2257,  0.0288,  0.2546,  0.4107,
     0.4756,  0.4419,  0.3242,  0.1609,  0.0000,  0.0000, -0.1273,  0.0000,  0.0000
};

// =================================================================================
// NEW: STANDALONE FIR DECIMATOR FUNCTION DEFINITION
// This function contains the logic that was previously trapped inside ddc_demodulator.
// It can now be called directly from circuit_final.cpp.
// =================================================================================
void fir_decim_ddc(
    data_t sample_in,
    baseband_t& sample_out,
    filter_accum_t state[FIR_TAPS],
    bool& data_valid
) {
    #pragma HLS INLINE // Suggests the compiler to inline this for performance
    #pragma HLS ARRAY_PARTITION variable=state complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lpf_coeffs complete dim=1

    static int decim_count = 0;

    // Shift samples through the delay line (filter state)
    for (int i = FIR_TAPS - 1; i > 0; i--) {
        #pragma HLS UNROLL
        state[i] = state[i - 1];
    }
    state[0] = sample_in;

    // Check if it's time to produce an output
    if (decim_count == 0) {
        filter_accum_t acc = 0;
        // Compute the FIR filter output (convolution)
        for (int i = 0; i < FIR_TAPS; i++) {
            #pragma HLS UNROLL
            acc += state[i] * lpf_coeffs[i];
        }
        sample_out = acc;
        data_valid = true; // Signal that the output is valid
    } else {
        data_valid = false; // No valid output on this clock cycle
    }

    // Update the decimation counter
    decim_count = (decim_count + 1) % DECIM_FACTOR;
}

// NEW: DUAL-PATH FIR DECIMATOR FUNCTION DEFINITION
// This function processes I and Q together, using one safe static counter.
// =================================================================================
void fir_decim_dual_path(
    data_t i_sample_in,
    data_t q_sample_in,
    baseband_t& i_sample_out,
    baseband_t& q_sample_out,
    filter_accum_t i_state[FIR_TAPS],
    filter_accum_t q_state[FIR_TAPS],
    bool& data_valid
) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=i_state complete dim=1
    #pragma HLS ARRAY_PARTITION variable=q_state complete dim=1
    #pragma HLS ARRAY_PARTITION variable=lpf_coeffs complete dim=1

    static int decim_count = 0; // This is now safe as it's only used once per loop

    // Shift samples for I-path
    for (int i = FIR_TAPS - 1; i > 0; i--) {
    	#pragma HLS UNROLL
        i_state[i] = i_state[i - 1];
    }
    i_state[0] = i_sample_in;

    // Shift samples for Q-path
    for (int i = FIR_TAPS - 1; i > 0; i--) {
    	#pragma HLS UNROLL
        q_state[i] = q_state[i - 1];
    }
    q_state[0] = q_sample_in;

    // Check if it's time to produce an output
    if (decim_count == 0) {
        filter_accum_t i_acc = 0;
        filter_accum_t q_acc = 0;

        // Compute FIR for I-path
        for (int i = 0; i < FIR_TAPS; i++) {
        	#pragma HLS UNROLL
            i_acc += i_state[i] * lpf_coeffs[i];
        }
        // Compute FIR for Q-path
        for (int i = 0; i < FIR_TAPS; i++) {
        	#pragma HLS UNROLL
            q_acc += q_state[i] * lpf_coeffs[i];
        }

        i_sample_out = i_acc;
        q_sample_out = q_acc;
        data_valid = true;
    } else {
        data_valid = false;
    }

    // Update the decimation counter
    decim_count = (decim_count + 1) % DECIM_FACTOR;
}


// =================================================================================
// ORIGINAL DDC FUNCTION (REFACTORED TO USE THE NEW FIR FUNCTION)
// This remains for completeness but is no longer directly used by circuit_final's feedback path.
// =================================================================================
void ddc_demodulator(
    rf_sample_t rf_in[],
    baseband_t i_out[],
    baseband_t q_out[],
    int num_samples,
    ap_uint<32> freq_word,
    ap_fixed<16,8> gain
) {
    // NCO and LUTs for mixing
    static ap_uint<32> nco_phase = 0;
    const filter_accum_t sine_lut[1024] = {
    		#include "sin_lut.h"
};
    const filter_accum_t cosine_lut[1024] = {
    		#include "cos_lut.h"
};

    // State for the internal FIR filters
    static i_fir_state_t i_fir_state;
    static q_fir_state_t q_fir_state;

    int out_sample_idx = 0;

    // FIXED: Calculate the expected number of output samples for bounds checking
    const int expected_output_samples = num_samples / DECIM_FACTOR;

    for (int n = 0; n < num_samples; ++n) {
        #pragma HLS PIPELINE

        // --- NCO and Mixer Stage ---
        nco_phase += freq_word;
        ap_uint<10> table_idx = nco_phase >> 22;
        rf_sample_t cos_val = cosine_lut[table_idx];
        rf_sample_t sin_val = sine_lut[table_idx];
        data_t i_mixed = rf_in[n] * cos_val;
        data_t q_mixed = rf_in[n] * (-sin_val);

        // --- FIR Decimation Stage ---
        bool i_valid, q_valid;
        baseband_t i_filtered, q_filtered;

        // Call the new, standalone FIR function for each path
        fir_decim_ddc(i_mixed, i_filtered, i_fir_state, i_valid);
        fir_decim_ddc(q_mixed, q_filtered, q_fir_state, q_valid);

        // If the filters produced a valid output, apply gain and store it
        if (i_valid && q_valid) {
            // FIXED: Use the calculated bound instead of the missing macro
            if (out_sample_idx < expected_output_samples) {
                i_out[out_sample_idx] = i_filtered * gain;
                q_out[out_sample_idx] = q_filtered * gain;
                out_sample_idx++;
            }
        }
    }
}
