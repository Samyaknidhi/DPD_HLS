#ifndef DPD_2_H
#define DPD_2_H

#include "ap_fixed.h"

// CHANGE THIS: Reduce model complexity
#define K 13 // Nonlinearity order

#ifndef MEMORY_DEPTH
#define MEMORY_DEPTH 12  // Increased from 5 to handle the significant delay
#endif

// Data types for DPD system
typedef ap_fixed<24,8> data_t;    // Input/output data type (I/Q)
typedef ap_fixed<24,8> coef_t;    // Coefficient type for LMS
typedef ap_fixed<24,6> phi_t;     // Basis function output type
typedef ap_fixed<32,8> acc_t;     // Accumulator type for MAC

// Complex coefficient struct
typedef struct {
    coef_t real;
    coef_t imag;
} ccoef_t;

// DPD function prototype (matches new dpd.cpp interface)
// Memory-based DPD function prototype
void dpd(
    const data_t i_in[MEMORY_DEPTH], const data_t q_in[MEMORY_DEPTH],
    data_t i_ref, data_t q_ref,
    ccoef_t w[K][MEMORY_DEPTH], coef_t mu,
    data_t* z_i, data_t* z_q
);

// Basis function computation
void compute_phi_all(
    data_t i, data_t q,
    phi_t real_phi[K], phi_t imag_phi[K]
);

#endif // DPD_2_H
