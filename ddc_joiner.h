#ifndef DDC_JOINER_H
#define DDC_JOINER_H

#include <ap_fixed.h>
#include <ap_int.h>

// Configuration parameters
#define DECIM_FACTOR 8
#define FIR_TAPS 63

typedef ap_fixed<16,8> rf_sample_t;       // 16-bit RF input samples
typedef ap_fixed<18,2> filter_coeff_t;    // Filter coefficient type
typedef ap_fixed<24,8> filter_accum_t;    // Accumulator with growth for FIR filters
typedef ap_fixed<24,8> baseband_t;        // Baseband I/Q output samples

void ddc_demodulator(
    rf_sample_t rf_in[],
    baseband_t i_out[],
    baseband_t q_out[],
    int num_samples,
    ap_uint<32> freq_word,
    ap_fixed<16,8> gain
);

#endif // DDC_JOINER_H
