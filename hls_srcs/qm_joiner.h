#ifndef QM_JOIN_H
#define QM_JOIN_H

#include <ap_fixed.h>
#include <ap_int.h>

typedef ap_fixed<24,8> data_t;

// NCO using ap_fixed and LUT approach
void nco(ap_uint<8> &phase, ap_uint<8> phase_inc, data_t &cos_lo, data_t &sin_lo);

// Quadrature modulator: s(t) = I*cos(lo) - Q*sin(lo)
data_t digital_qm(data_t I, data_t Q, data_t cos_lo, data_t sin_lo);

#endif
