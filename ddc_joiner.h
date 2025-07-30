#ifndef DDC_JOINER_H
#define DDC_JOINER_H

#include <ap_fixed.h>
#include <ap_int.h>

// Consistent type definitions
#define FIR_TAPS 63
#define DECIM_FACTOR 8 // Make sure this is defined for ddc_join.cpp
typedef ap_fixed<24,8> baseband_t;
typedef ap_fixed<16,8> rf_sample_t;
typedef ap_fixed<24,8> data_t;
typedef ap_fixed<24,8> filter_accum_t;

// Filter state types
typedef filter_accum_t i_fir_state_t[FIR_TAPS];
typedef filter_accum_t q_fir_state_t[FIR_TAPS];

// Original single-path FIR prototype (keep for modularity)
void fir_decim_ddc(
    data_t sample_in,
    baseband_t& sample_out,
    filter_accum_t state[FIR_TAPS],
    bool& data_valid
);

// ===================================================================
// NEW: DUAL-PATH FIR DECIMATOR PROTOTYPE
// This is the new function that circuit_final will call to avoid the static variable bug.
// ===================================================================
void fir_decim_dual_path(
    data_t i_sample_in,
    data_t q_sample_in,
    baseband_t& i_sample_out,
    baseband_t& q_sample_out,
    filter_accum_t i_state[FIR_TAPS],
    filter_accum_t q_state[FIR_TAPS],
    bool& data_valid
);


// Original DDC demodulator prototype
void ddc_demodulator(
    rf_sample_t rf_in[],
    baseband_t i_out[],
    baseband_t q_out[],
    int num_samples,
    ap_uint<32> freq_word,
    ap_fixed<16,8> gain
);

#endif // DDC_JOINER_H
