#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<24,8> data_ty;

// Multi-bit DAC with channel select (8-bit output)
void dac_multibit_with_select(data_ty din, ap_fixed<24,8> &dout, bool channel_select) {
#pragma HLS INLINE
    // SOLUTION: Remove scaling and clamping.
    // The DPD output is already correctly scaled. Just pass it through.
    dout = din;
    // Optional: Keep debug print
    printf( "dout value is %f", dout.to_float());
}
