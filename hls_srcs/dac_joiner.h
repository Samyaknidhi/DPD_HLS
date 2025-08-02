#ifndef DAC_JOIN_H
#define DAC_JOIN_H

#include <ap_fixed.h>
#include <ap_int.h>

// Use the same data_t as your DPD output
typedef ap_fixed<24,8> data_ty;

// Multi-bit DAC with channel select (8-bit output)
void dac_multibit_with_select(data_ty din, ap_fixed<24,8> &dout, bool channel_select);

#endif
