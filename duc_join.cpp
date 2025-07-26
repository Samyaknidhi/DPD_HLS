#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"
#include <math.h>
#include "duc_joiner.h"

// Configuration parameters
#define INTERPOLATION_FACTOR 8
#define NCO_LUT_SIZE 1024
#define NCO_LUT_BITS 10
#define NUM_FILTER_TAPS 63

// Fixed-point type definitions
typedef ap_fixed<24,8> sample_type;
typedef ap_fixed<24,12> coeff_type;
typedef ap_fixed<24,12> acc_type;
typedef ap_uint<12> phase_type;

// FIXED: Use your existing filter coefficients (they look correct)
const coeff_type FILTER_COEFFS[NUM_FILTER_TAPS] = {
    0.00156, 0.00258, 0.00376, 0.00511, 0.00662, 0.00828, 0.01007, 0.01196, 0.01392,
    0.01594, 0.01796, 0.01997, 0.02193, 0.02379, 0.02553, 0.02713, 0.02855, 0.02978,
    0.03079, 0.03157, 0.03209, 0.03235, 0.03235, 0.03209, 0.03157, 0.03079, 0.02978,
    0.02855, 0.02713, 0.02553, 0.02379, 0.02193, 0.01997, 0.01796, 0.01594, 0.01392,
    0.01196, 0.01007, 0.00828, 0.00662, 0.00511, 0.00376, 0.00258, 0.00156, 0.00069,
    0.00000, -0.00053, -0.00091, -0.00113, -0.00123, -0.00122, -0.00113, -0.00098,
    -0.00079, -0.00058, -0.00038, -0.00020, -0.00006, 0.00003, 0.00009, 0.00010, 0.00009,
    0.00006
};

// FIXED: Use your existing LUT files
const sample_type sine_lut[NCO_LUT_SIZE] = {
    #include "sin_lut.h"  // Your existing file
};
const sample_type cosine_lut[NCO_LUT_SIZE] = {
    #include "cos_lut.h"  // Your existing file
};

// FIXED: Proper polyphase interpolating filter
void polyphase_interpolate(
    sample_type input_sample,
    sample_type shift_reg[NUM_FILTER_TAPS],
    sample_type output_samples[INTERPOLATION_FACTOR]
) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output_samples complete dim=1

    // Shift register update (only when new sample arrives)
    for (int i = NUM_FILTER_TAPS-1; i > 0; i--) {
        #pragma HLS UNROLL
        shift_reg[i] = shift_reg[i-1];
    }
    shift_reg[0] = input_sample;

    // Generate INTERPOLATION_FACTOR outputs using polyphase decomposition
    for (int phase = 0; phase < INTERPOLATION_FACTOR; phase++) {
        #pragma HLS UNROLL
        acc_type acc = 0;

        // Each polyphase filter uses every 8th coefficient
        for (int i = phase; i < NUM_FILTER_TAPS; i += INTERPOLATION_FACTOR) {
            #pragma HLS PIPELINE
            acc += shift_reg[i] * FILTER_COEFFS[i];
        }
        output_samples[phase] = sample_type(acc) * INTERPOLATION_FACTOR;
    }
}

// FIXED: Batch processing DUC for proper circuit_final integration
void digital_upconverter_batch(
    sample_type i_samples[DATA_LEN],
    sample_type q_samples[DATA_LEN],
    sample_type output[DATA_LEN * INTERPOLATION_FACTOR],
    ap_uint<32> freq_control_word,
    ap_uint<1> enable
) {
    #pragma HLS INTERFACE s_axilite port=freq_control_word bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=enable bundle=CTRL
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS ARRAY_PARTITION variable=FILTER_COEFFS complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sine_lut cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=cosine_lut cyclic factor=8 dim=1

    // Static variables for persistent state
    static sample_type i_shift_reg[NUM_FILTER_TAPS] = {0};
    static sample_type q_shift_reg[NUM_FILTER_TAPS] = {0};
    static ap_uint<32> phase_acc = 0;  // Use 32-bit phase accumulator

    if (!enable) return;

    int output_idx = 0;

    // Process each input sample
    for (int n = 0; n < DATA_LEN; n++) {
        #pragma HLS PIPELINE II=8  // Process 8 outputs per input

        // Interpolate both I and Q channels
        sample_type i_interp[INTERPOLATION_FACTOR];
        sample_type q_interp[INTERPOLATION_FACTOR];

        polyphase_interpolate(i_samples[n], i_shift_reg, i_interp);
        polyphase_interpolate(q_samples[n], q_shift_reg, q_interp);

        // Generate INTERPOLATION_FACTOR upconverted samples
        for (int k = 0; k < INTERPOLATION_FACTOR; k++) {
            #pragma HLS UNROLL

            // Update NCO phase
            phase_acc += freq_control_word;

            // FIXED: Proper LUT indexing with your files
            ap_uint<NCO_LUT_BITS> lut_addr = phase_acc >> (32 - NCO_LUT_BITS);
            sample_type sin_val = sine_lut[lut_addr];
            sample_type cos_val = cosine_lut[lut_addr];

            // Complex upconversion: I*cos - Q*sin
            sample_type upconverted = i_interp[k] * cos_val - q_interp[k] * sin_val;

            if (output_idx < DATA_LEN * INTERPOLATION_FACTOR) {
                output[output_idx++] = upconverted;
            }
        }

        // Debug for first few samples
        #ifndef __SYNTHESIS__
        if (n < 3) {
            printf("DUC[%d]: i_in=%f, q_in=%f, phase=0x%08X, interp[0]=%f, out=%f\n",
                   n, i_samples[n].to_double(), q_samples[n].to_double(),
                   (unsigned int)phase_acc, i_interp[0].to_double(),
                   (output_idx > 0) ? output[output_idx-1].to_double() : 0.0);
        }
        #endif
    }
}

// FIXED: Streaming DUC (corrected interpolation logic)
void digital_upconverter(
    hls::stream<sample_type> &i_in,
    hls::stream<sample_type> &q_in,
    hls::stream<sample_type> &signal_out,
    ap_uint<32> freq_control_word,
    ap_uint<1> enable
) {
    #pragma HLS INTERFACE axis port=i_in
    #pragma HLS INTERFACE axis port=q_in
    #pragma HLS INTERFACE axis port=signal_out
    #pragma HLS INTERFACE s_axilite port=freq_control_word bundle=CTRL
    #pragma HLS INTERFACE s_axilite port=enable bundle=CTRL
    #pragma HLS INTERFACE ap_ctrl_none port=return
    #pragma HLS ARRAY_PARTITION variable=FILTER_COEFFS complete dim=1
    #pragma HLS ARRAY_PARTITION variable=sine_lut cyclic factor=4 dim=1
    #pragma HLS ARRAY_PARTITION variable=cosine_lut cyclic factor=4 dim=1

    // Static variables for persistent state
    static sample_type i_shift_reg[NUM_FILTER_TAPS] = {0};
    static sample_type q_shift_reg[NUM_FILTER_TAPS] = {0};
    static ap_uint<32> phase_acc = 0;

    if (!enable) return;

    // Read one input sample
    sample_type i_sample = 0, q_sample = 0;
    if (!i_in.empty() && !q_in.empty()) {
        i_in >> i_sample;
        q_in >> q_sample;
    } else {
        return;
    }

    // FIXED: Proper polyphase interpolation
    sample_type i_interp[INTERPOLATION_FACTOR];
    sample_type q_interp[INTERPOLATION_FACTOR];

    polyphase_interpolate(i_sample, i_shift_reg, i_interp);
    polyphase_interpolate(q_sample, q_shift_reg, q_interp);

    // Generate interpolated and upconverted outputs
    for (int k = 0; k < INTERPOLATION_FACTOR; k++) {
        #pragma HLS PIPELINE II=1

        // Update phase accumulator
        phase_acc += freq_control_word;

        // FIXED: Correct LUT indexing
        ap_uint<NCO_LUT_BITS> lut_addr = phase_acc >> (32 - NCO_LUT_BITS);
        sample_type sin_val = sine_lut[lut_addr];
        sample_type cos_val = cosine_lut[lut_addr];

        // Complex upconversion
        sample_type upconverted = i_interp[k] * cos_val - q_interp[k] * sin_val;
        signal_out << upconverted;
    }
}

// Simple FIR filter (keep for backward compatibility)
void fir_filter(
    sample_type shift_reg[NUM_FILTER_TAPS],
    sample_type input_sample,
    sample_type &output_sample
) {
    #pragma HLS INLINE

    // Shift the register
    for (int i = NUM_FILTER_TAPS-1; i > 0; i--) {
        #pragma HLS UNROLL
        shift_reg[i] = shift_reg[i-1];
    }
    shift_reg[0] = input_sample;

    // Calculate filter output
    acc_type acc = 0;
    for (int i = 0; i < NUM_FILTER_TAPS; i++) {
        acc += shift_reg[i] * FILTER_COEFFS[i];
    }
    output_sample = acc;
}
