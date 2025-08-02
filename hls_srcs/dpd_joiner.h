#ifndef DPD_JOINER_H
#define DPD_JOINER_H

#include "ap_fixed.h"

// --- Model Parameters ---
#define K 5             // Nonlinearity order
#define MEMORY_DEPTH 2  // Number of memory taps

// --- Fixed-Point Type Definitions (CORRECTED FOR STABILITY) ---

// Input/output data type (I/Q) for the DPD module
typedef ap_fixed<24, 8> data_t;

// Basis function output type (|z|^2). Must hold i*i + q*q.
// The integer part must be >= I(data_t)*2. Here, 8*2=16.
typedef ap_fixed<32, 16> phi_t;

// Coefficient type for the filter weights (w) and step-size (mu)
typedef ap_fixed<32, 2> coef_t;

// Accumulator for the MAC operation (sum(w*phi)).
// Integer part must be >= I(w) + I(phi) + log2(K*M).
// 2 + 16 + log2(21) ~ 18 + 5 = 23. We use 24 for safety.
typedef ap_fixed<48, 24> acc_t;

// CRITICAL FIX: Intermediate type for the weight update calculation (err * phi).
// This was the source of a major overflow bug.
// Integer part must be >= I(err) + I(phi) = I(data_t) + I(phi) = 8 + 16 = 24.
typedef ap_fixed<48, 24> update_t;


// --- Complex Coefficient Struct ---
typedef struct {
    coef_t real;
    coef_t imag;
} ccoef_t;


// --- Function Prototype ---
// The function name from your dpd_join.cpp is dpd(), not dpd_join()
void dpd(
    const data_t i_in[MEMORY_DEPTH],
    const data_t q_in[MEMORY_DEPTH],
    data_t i_ref,
    data_t q_ref,
    ccoef_t w[K][MEMORY_DEPTH],
    coef_t mu,
    data_t *i_out,
    data_t *q_out
);

#endif // DPD_JOINER_H
