#include <ap_int.h>
#include <ap_fixed.h>

typedef ap_fixed<24,8> fixed_point_t;

void conste(
    ap_fixed<16,8> input_bytes[512],
    int num_bits,
    fixed_point_t output_symbols_I[2048],
    fixed_point_t output_symbols_Q[2048]
);
