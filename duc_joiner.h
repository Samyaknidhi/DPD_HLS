#ifndef DIGITALUC_H
#define DIGITALUC_H

#include "ap_int.h"
#include "ap_fixed.h"
#include "hls_stream.h"

// Configuration parameters (should match your .cpp)
#define INTERPOLATION_FACTOR 8
#define NCO_LUT_SIZE 1024
#define NCO_LUT_BITS 10
#define NUM_FILTER_TAPS 63
#define DATA_LEN 512

typedef ap_fixed<24,8> sample_type;
typedef ap_fixed<24,12> coeff_type;
typedef ap_fixed<24,12> acc_type;
typedef ap_uint<12> phase_type;

void digital_upconverter_batch(
    sample_type i_samples[DATA_LEN],
    sample_type q_samples[DATA_LEN],
    sample_type output[DATA_LEN * INTERPOLATION_FACTOR],
    ap_uint<32> freq_control_word,
    ap_uint<1> enable
);

// Top-level DUC function prototype
void digital_upconverter(
    hls::stream<sample_type> &i_in,
    hls::stream<sample_type> &q_in,
    hls::stream<sample_type> &signal_out,
    ap_uint<32> freq_control_word,
    ap_uint<1> enable
);

#endif // DIGITALUC_H
