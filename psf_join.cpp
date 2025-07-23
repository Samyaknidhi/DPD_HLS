#include "ap_fixed.h"
#include "hls_math.h"

#define DATA_LEN 8192
#define NUM_WEIGHTS 41
#define SPS 25
#define ALPHA 0.5

typedef ap_fixed<24, 8> fixed_t;

void raised_cosine_filter(fixed_t rc[NUM_WEIGHTS]) {
    const fixed_t PI = fixed_t(3.14159);
    const fixed_t ALPHA_FIXED = fixed_t(ALPHA);
    const fixed_t EPS = fixed_t(1e-5);
    const fixed_t EPS_X = fixed_t(1e-3);
    const fixed_t SCALE = fixed_t(0.9999);

    int mid = NUM_WEIGHTS / 2;
    fixed_t sum = fixed_t(0.001);  // <- Keep your initialization

    for (int i = 0; i < NUM_WEIGHTS; i++) {
#pragma HLS PIPELINE II=1  // <- Add this for performance
        fixed_t idx = fixed_t(i - mid);
        fixed_t x = SCALE * idx / fixed_t(SPS);
        fixed_t pi_x = PI * x;

        fixed_t sinc;
        if (hls::abs(x) < EPS_X)
            sinc = fixed_t(1.0);
        else
            sinc = hls::sin(pi_x) / pi_x;

        fixed_t denom = fixed_t(1.0) - fixed_t(4.0) * ALPHA_FIXED * ALPHA_FIXED * x * x;

        // FIXED: Better denominator safety
        if (hls::abs(denom) < EPS) {
            denom = (denom >= fixed_t(0.0)) ? EPS : fixed_t(-EPS);  // Preserve sign
        }

        // FIXED: Use consistent type for angle
        fixed_t angle = PI * ALPHA_FIXED * x;  // <-Changed from ap_fixed<32,6>
        fixed_t cos_part = hls::cos(angle);

        rc[i] = sinc * (cos_part / denom);
        sum += rc[i];
    }

    // Keep your normalization approach (peak normalization)
    fixed_t max_abs = fixed_t(0.0);  // <- Initialize properly
    for (int i = 0; i < NUM_WEIGHTS; i++) {
#pragma HLS PIPELINE II=1
        if (hls::abs(rc[i]) > max_abs)
            max_abs = hls::abs(rc[i]);
    }

    if (hls::abs(max_abs) < EPS)
        max_abs = fixed_t(1.0);

    for (int i = 0; i < NUM_WEIGHTS; i++) {
#pragma HLS PIPELINE II=1
        rc[i] = rc[i] / max_abs;
    }
}

void convolve(const fixed_t data[DATA_LEN], const fixed_t filter[NUM_WEIGHTS], fixed_t result[DATA_LEN]) {
#pragma HLS ARRAY_PARTITION variable=filter complete dim=1  // <- Add this
    int mid = NUM_WEIGHTS / 2;

    for (int i = 0; i < DATA_LEN; i++) {
#pragma HLS PIPELINE II=1  // <- Add this
        fixed_t acc = fixed_t(0.0);  // <- Proper initialization
        for (int j = 0; j < NUM_WEIGHTS; j++) {
#pragma HLS UNROLL factor=4  // <- Add this
            int k = i - mid + j;
            if (k >= 0 && k < DATA_LEN)
                acc += data[k] * filter[j];
        }
        result[i] = acc;
    }
}

void pulse_shape(fixed_t i_data[DATA_LEN], fixed_t q_data[DATA_LEN],
                 fixed_t i_out[DATA_LEN], fixed_t q_out[DATA_LEN]) {
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL
#pragma HLS INTERFACE bram port=i_data
#pragma HLS INTERFACE bram port=q_data
#pragma HLS INTERFACE bram port=i_out
#pragma HLS INTERFACE bram port=q_out

    fixed_t rc_filter[NUM_WEIGHTS];
#pragma HLS ARRAY_PARTITION variable=rc_filter complete dim=1  // <- Add this

    raised_cosine_filter(rc_filter);
    convolve(i_data, rc_filter, i_out);
    convolve(q_data, rc_filter, q_out);
}
