#include <ap_fixed.h>

typedef ap_fixed<24,8> fixed_t;

void pulse_shape(
    fixed_t i_data[512],
    fixed_t q_data[512],
    fixed_t i_out[512],
    fixed_t q_out[512]
);
